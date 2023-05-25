import os
from argparse import ArgumentParser, Namespace
from functools import partial
from typing import Any, Dict, Iterator, List, Tuple

import matplotlib as mpl  # type: ignore
import numpy as np
import pandas as pd  # type: ignore
import torch
from base import Algorithm, HashableModule
from data.get import get_dataset
from matplotlib import animation  # type: ignore
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation  # type: ignore
from scipy.stats import multivariate_normal  # type: ignore
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset
from tqdm import tqdm  # type: ignore
from utils import Stats, md5, seed, set_logger, str2bool

from umbc.models.mvn import (MVNSSE, MVNSSEUMBC, MVNDeepSets, MVNDiffEM,
                                   MVNDiffEMUMBC, MVNFSPool, MVNFSPoolUMBC,
                                   MVNSetTransformer, MVNSSEHierarchical)

T = torch.Tensor
SetEncoder = nn.Module


class MVNTrainer(Algorithm):
    def __init__(
        self,
        args: Namespace,
        model: nn.Module,
        optimizer: Any,
        dataset: Dataset
    ):
        super().__init__()

        self.args = args
        self.model = model.to(self.args.device)

        self.optimizer = optimizer
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(
            self.optimizer,
            milestones=args.decay_epochs,
            gamma=args.decay_gamma
        )
        self.trainset = dataset
        self.best_nll = float("inf")
        self.epoch = 0
        self.finished = False
        self.loaded = False
        self.tuned = False

        addendum = ""
        self.results_path = os.path.join(
            "results",
            addendum,
            f"{dataset.name}",  # type: ignore
            f"{self.model.name}"
        )
        self.models_path = os.path.join(self.results_path, "models")
        for d in [self.results_path, self.models_path]:
            os.makedirs(d, exist_ok=True)

        # write the model string to file under the model hash so we will
        # always know which model created this hash
        path = os.path.join(self.models_path, f"{self.args.model_hash}.txt")
        with open(path, "w") as f:
            f.write(self.args.model_string)

        self.tr_stats = Stats(["loss"])
        self.full_te_stats = [Stats(["loss"])
                              for _ in self.args.test_set_sizes]
        self.sub_te_stats = [Stats(["loss"]) for _ in self.args.test_set_sizes]

    def fit(self) -> None:
        self.load_model(self.models_path)
        if self.finished:
            self.log("called fit() on a model which has finished training")
            return

        while self.epoch < self.args.epochs:
            # re-initialize sigma and lambda here because during fitting we
            # want to make sure they are blank for the validation phase,
            # but they should be saved in the current model for future
            # testing
            {"train": self.train}[self.args.mode]()
            self.scheduler.step()
            self.log_train_stats(self.results_path)

            do_val = self.args.model not in [
                "diff-em", "fspool", "diff-em-umbc", "fspool-umbc"]
            if self.epoch % 5 == 0 and do_val:
                self.test(its=1000)
                self.log_test_stats(self.results_path, test_name="val")

            self.save_model(self.models_path)
            self.epoch += 1

        self.log("finished training, saving...")
        self.save_model(self.models_path, finished=True)

    def forward(self, x: T) -> T:
        if self.args.grad_set_size == self.args.train_set_size:
            return self.model(x)  # type: ignore

        if not model.is_mbc:
            raise ValueError("partitioned forward is only for MBC")

        idx = torch.randperm(x.size(1))
        x_grad, x_nograd = x[:, idx[:self.args.grad_set_size]
                             ], x[:, idx[self.args.grad_set_size:]]

        return self.model.partitioned_forward(x_grad, x_nograd)  # type: ignore

    def train(self) -> None:
        self.model.train()
        t = tqdm(range(self.args.epoch_its), ncols=75, leave=False)

        for i in t:
            b, s = self.args.batch_size, self.args.train_set_size
            c = self.args.classes
            x, _, _, _, _ = self.trainset.sample(b, s, c)  # type: ignore
            x = x.to(self.args.device)

            out = self.forward(x)
            ll, _ = self.trainset.log_prob(  # type: ignore
                x, *self.trainset.parse(out))  # type: ignore
            loss = -ll

            self.optimizer.zero_grad()
            loss.backward()
            gs_diff = (self.args.grad_set_size != self.args.train_set_size)
            if gs_diff and self.args.grad_correction and self.model.is_mbc:
                self.model.grad_correct(
                    self.args.train_set_size / self.args.grad_set_size)  # type: ignore  # noqa

            t.set_description(f"{loss.item()=:.4f}")
            self.optimizer.step()
            with torch.no_grad():
                self.tr_stats.update_loss(loss * x.size(1), x.size(1))

    def test_forward(self, x: T) -> T:
        out = []
        self.model.pre_forward_mbc()  # type: ignore

        x_t, c_t = None, None  # type: ignore
        for x_in in x.split(self.args.grad_set_size, dim=1):
            x_t, c_t = self.model.forward_mbc(x_in, X_prev=x_t, c_prev=c_t)
        r = self.model.post_forward_mbc(x_t, c=c_t)
        out.append(self.model.decoder(r))

        return torch.cat(out)  # type: ignore

    def test(self, its: int) -> None:  # type: ignore
        self.model.eval()
        t = tqdm(range(its), ncols=75, leave=False)
        for it in t:
            x, y, pi, mu, sigma = self.trainset.sample(  # type: ignore
                self.args.batch_size,
                self.args.test_set_size,
                self.args.classes
            )
            x, y, pi, mu, sigma = map(lambda v: v.to(  # type: ignore
                self.args.device), (x, y, pi, mu, sigma))

            with torch.no_grad():
                for i, ss in enumerate(self.args.test_set_sizes):
                    xss = x[:, :ss]
                    out = self.test_forward(xss)

                    full_ll, yhat = self.trainset.log_prob(  # type: ignore
                        x, *self.trainset.parse(out))  # type: ignore
                    self.full_te_stats[i].update_loss(
                        -full_ll * y.numel(), y.numel())

                    sub_ll, yhat = self.trainset.log_prob(  # type: ignore
                        xss, *self.trainset.parse(out))  # type: ignore
                    self.sub_te_stats[i].update_loss(
                        -sub_ll * y.numel(), y.numel())

    def get_sampling_strategies(self) -> List[Any]:
        # define our sampling strategies. We will need to go through the batch
        # one by one and yield different set sample for each set. This can
        # then be iterated over for each set
        def single_point_stream(x: T, y: T) -> Iterator[Tuple[T, T]]:
            """simply sample one element of each set until we reach them end"""
            for idx in torch.randperm(x.size(0)):
                yield x[idx].unsqueeze(0), y[idx].unsqueeze(0)

        def class_stream(x: T, y: T) -> Iterator[Tuple[T, T]]:
            """return each class as a single sample"""
            for cl in y.unique():
                idx = y == cl
                yield x[idx], y[idx]

        def get_chunk_stream(n: int) -> Any:
            def chunk_stream(x: T, y: T) -> Iterator[Tuple[T, T]]:
                """
                simply sample one element of each set until we reach them end
                """
                for idx in torch.randperm(x.size(0)).chunk(n):
                    yield x[idx], y[idx]

            return chunk_stream

        def one_each_stream(x: T, y: T) -> Iterator[Tuple[T, T]]:
            """
            separate each class and then return one instanc of each class
            until all instances are covered. classes will have different
            numbers of instances, so if oen class runs out of examples,
            just return the other classes
            """
            xy_lst: List[Tuple[T, T]] = []
            for cl in y.unique():
                idx = y == cl
                xy_lst.append((x[idx], y[idx]))

            max_size = max([v[0].size(0) for v in xy_lst])
            for i in range(max_size):
                out_x, out_y = [], []
                for _x, _y in xy_lst:
                    if i > _x.size(0) - 1:
                        continue
                    out_x.append(_x[i])
                    out_y.append(_y[i])
                yield torch.stack(out_x), torch.stack(out_y)

        return [
            single_point_stream,
            class_stream,
            one_each_stream,
            get_chunk_stream(128)
        ]

    def motivation_example(
        self,
        x: T,
        y: T,
        pi: T,
        mu: T,
        sigma: T,
        its: int = 1
    ) -> None:  # type: ignore
        self.model.eval()
        mpl.use('Agg')
        mpl.rcParams.update(mpl.rcParamsDefault)

        outpath = os.path.join(
            "results",
            self.trainset.name,  # type: ignore
            "plots",
            "motivation"
        )
        os.makedirs(outpath, exist_ok=True)

        def figsize(scale, nplots=1):
            # Get this from LaTeX using \the\textwidth
            fig_width_pt = 390.0
            inches_per_pt = 1.0/72.27                       # Convert pt to inch
            # Aesthetic ratio (you could change this)
            golden_mean = (np.sqrt(5.0)-1.0)/2.0
            fig_width = fig_width_pt*inches_per_pt*scale    # width in inches
            fig_height = nplots*fig_width*golden_mean              # height in inches
            fig_size = [fig_width, fig_height]
            return fig_size

        pgf_with_latex = {                      # setup matplotlib to use latex for output
            "pgf.texsystem": "pdflatex",        # change this if using xetex or lautex
            "text.usetex": True,                # use LaTeX to write all text
            "font.family": "Times New Roman",
            "font.sans-serif": [],
            "font.size": 12,
            # default fig size of 0.9 textwidth
            "figure.figsize": figsize(1.0),
            "pgf.preamble": [
                # use utf8 fonts becasue your computer can handle it :)
                r"\usepackage[utf8x]{inputenc}",
                # plots will be generated using this preamble
                r"\usepackage[T1]{fontenc}",
            ]
        }
        mpl.rcParams.update(pgf_with_latex)

        sampling_strategies = self.get_sampling_strategies()
        stats = [Stats(["loss"]) for _ in sampling_strategies]

        x, y, pi, mu, sigma = map(lambda v: v.to(
            self.args.device), (x, y, pi, mu, sigma))
        for n, (sx, sy) in enumerate(zip(x, y)):
            xlimit = (sx[:, 0].amin().cpu() - 1, sx[:, 0].amax().cpu() + 1)
            ylimit = (sx[:, 1].amin().cpu() - 1, sx[:, 1].amax().cpu() + 1)

            # if n < 2:
            #     continue
            # if n == 3:
            #     return
            samples = [
                2, 54, 68, 71, 81, 98,
                # 103, 109, 117, 122, 149, 168, 175
            ]
            if n not in samples:
                continue
            if n == 200:
                return

            sd = np.random.randint(0, 1000000)
            # reset the model for this mini batch processing. split it
            # into parts according to the sampling strategies, and save
            # the successive predictions and parts to make final plots
            # and gifs
            for strategy, _ in zip(sampling_strategies, stats):
                # type: ignore
                print(f"({n}) {self.model.name} {strategy.__name__}")
                full_out, x_t_outs, c_t_outs = torch.Tensor(), [], []  # type: ignore  # noqa
                x_parts, y_parts = [], []

                if full_out.numel() == 0:
                    seed(sd)
                    self.model.pre_forward_mbc()  # type: ignore
                    x_t, c_t = self.model.forward_mbc(
                        sx.unsqueeze(0))  # type: ignore
                    out = self.model.post_forward_mbc(
                        x_t, c=c_t)  # type: ignore
                    full_out = self.model.decoder(out).cpu()  # type: ignore

                seed(sd)
                self.model.pre_forward_mbc()  # type: ignore
                for part_x, part_y in strategy(sx, sy):
                    # add the batch dimension back in
                    def f(t: T) -> T:
                        return t.unsqueeze(0).to(self.args.device)
                    part_x, part_y = map(f, (part_x, part_y))

                    x_parts.append(part_x.squeeze(0))
                    y_parts.append(part_y.squeeze(0))

                    X_prev, c_prev = None, None
                    if len(x_t_outs) > 0 and len(c_t_outs) > 0:
                        X_prev = x_t_outs[-1].to(self.args.device)
                        c_prev = c_t_outs[-1].to(self.args.device)

                    x_t, c_t = self.model.forward_mbc(  # type: ignore
                        part_x, X_prev=X_prev, c_prev=c_prev, grad=False)

                    # deepsets was made to return an int, so wrap it in a T
                    if not isinstance(c_t, T):
                        c_t = torch.tensor(c_t)

                    x_t_outs.append(x_t.cpu())  # type: ignore
                    c_t_outs.append(c_t.cpu())

                for i, (x, c) in enumerate(zip(x_t_outs, c_t_outs)):
                    out = self.model.post_forward_mbc(
                        x.to(self.args.device),
                        c=c.to(self.args.device)
                    )  # type: ignore

                    x_t_outs[i] = self.model.decoder(out).cpu()  # type: ignore

                if full_out.numel() != 0:
                    print(
                        "diff between full and mbc " +
                        f"{strategy.__name__} {self.model.name}: " +
                        f"{torch.sum((full_out - x_t_outs[-1]).abs())}"
                    )

                # make a seaborn plot witht the last prediction and all the x
                # points as a scatterplot use the model_outs and x_parts
                # to make an animation of the performance at each step
                fig, ax = plt.subplots(1, 1, figsize=(8, 6))
                filename = f"{strategy.__name__}-{self.model.name}"
                data = pd.DataFrame({
                    "x": sx[:, 0].cpu().numpy().tolist(),
                    "y": sx[:, 1].cpu().numpy().tolist(),
                    "label": [str(v) for v in sy.cpu().numpy().tolist()],
                })

                for label, (color, marker) in enumerate(zip(
                    ["tab:red", "tab:blue", "tab:orange", "tab:green"],
                    ["o", "p", "P", "X"]
                )):
                    d = data[data.label == str(label)]
                    ax.scatter(
                        x=d["x"],
                        y=d["y"],
                        c=color,
                        marker=marker,
                        label=label,
                        edgecolor="white",
                        s=11**2
                    )

                outs = x_t_outs[-1].squeeze(0)
                _sx = sx.unsqueeze(0).cpu()
                ll, _ = self.trainset.log_prob(  # type: ignore
                    _sx, *self.trainset.parse(x_t_outs[-1]))  # type: ignore

                def contours(preds: Any, steps: int = 100) -> Any:
                    conts = []
                    for i, g in enumerate(preds):
                        stds = 20
                        mu, sigma = g[1:3], torch.clamp(
                            F.softplus(g[3:]), 1e-2)

                        mx = np.linspace(-sigma[0].item() *
                                         stds, sigma[0].item() * stds, steps)
                        my = np.linspace(-sigma[1].item() *
                                         stds, sigma[1].item() * stds, steps)
                        xx, yy = np.meshgrid(mx, my)

                        rv = multivariate_normal(
                            [0, 0],
                            [[sigma[0].item(), 0], [0, sigma[1].item()]]
                        )
                        data = np.dstack((xx, yy))
                        z = rv.pdf(data)
                        c = ax.contour(
                            mx + mu[0].cpu().numpy(),
                            my + mu[1].cpu().numpy(),
                            z,
                            5,
                            alpha=0.75,
                            linewidths=4.0,
                            cmap="inferno"
                        )
                        conts.append(c)
                    return conts

                contours(outs, steps=100)

                model_deref = {
                    "SetTransformer": "Set Transformer",
                    "SSEUMBC": "UMBC",
                    "SSE": "SSE",
                    "DeepSets": "DeepSets"
                }
                ax.set(xlabel="", ylabel="", xticks=[],
                       yticks=[], ylim=ylimit, xlim=xlimit)

                # type: ignore
                s = f"NLL ($\downarrow$): {-ll:.2f}"  # noqa
                ax.text(
                    x=-1.25, y=3.25, s=s,
                    fontsize=28, ha="center", va="center"
                )

                # s = f"{model_deref[self.model.name]}\n"  # type: ignore
                s = f"{' '.join(strategy.__name__.split('_'))}"
                ax.text(
                    x=-1.25, y=-2.7, s=s,
                    fontsize=28, ha="center", va="center"
                )

                ax.legend(fontsize=24, markerscale=2.0)

                i_outpath = os.path.join(outpath, str(n))
                print(i_outpath)
                os.makedirs(i_outpath, exist_ok=True)

                fig.tight_layout()
                fig.savefig(os.path.join(i_outpath, f"{filename}.pdf"))
                plt.clf()
                plt.cla()
                plt.close()

                # start the process of making an animation =============
                fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 6))
                model_name = self.model.name

                colors = ["tab:red", "tab:blue", "tab:orange", "tab:green"]
                markers = ["o", "p", "P", "X"]

                sc = [
                    ax.scatter(
                        x=[], y=[], c=c, marker=m, edgecolor="white", label=i, s=11**2
                    )
                    for i, (c, m) in enumerate(zip(colors, markers))
                ]

                x_parts = [v.cpu().numpy() for v in x_parts]
                x_parts_done = [[] for _ in colors]  # type: ignore

                class Animator:
                    def __init__(self, trainset: Dataset) -> None:
                        self.trainset = trainset

                    def update(self, frame: Any) -> Any:
                        i, ax = frame
                        print(".", end="", flush=True)

                        labels = y_parts[i].cpu().numpy()
                        for lbl in np.unique(labels):
                            idx = labels == lbl
                            # print(x_parts[:i + 1])

                            x_parts_done[lbl].append(x_parts[i][idx])
                            sc[lbl].set_offsets(
                                np.concatenate(x_parts_done[lbl]))
                            # sc[lbl].set_offsets(np.c_[x_parts[i][idx, 0].cpu().numpy(), x_parts[i][idx, 1].cpu().numpy()])  # noqa
                            # ax.scatter(x=x_parts[i][idx, 0].cpu().numpy(), y=x_parts[i][idx, 1].cpu().numpy(), c=colors[lbl], label=lbl, s=11**2)  # noqa

                        _sx = sx.unsqueeze(0).cpu()
                        ll, _ = self.trainset.log_prob(  # type: ignore
                            _sx, *self.trainset.parse(x_t_outs[i]))  # type: ignore  # noqa
                        if hasattr(self, "text"):
                            self.text.remove()  # type: ignore

                        # type: ignore
                        ax.set_title(
                            f"{model_deref[model_name]} " +  # type: ignore
                            f"({i}) {' '.join(strategy.__name__.split('_'))}",
                            fontsize=24
                        )

                        s = f"NLL$\downarrow$: {-ll:.2f}"  # noqa
                        self.text = ax.text(
                            x=-1.25, y=3.25, s=s,
                            fontsize=24, ha="center", va="center"
                        )  # type: ignore
                        ax.set(xlabel="", ylabel="", xticks=[],
                               yticks=[], xlim=xlimit, ylim=ylimit)

                        self.conts: Any
                        if hasattr(self, "conts"):
                            # type: ignore
                            for coll in [c.collections for c in self.conts]:
                                for tp in coll:
                                    tp.remove()

                        self.conts = contours(
                            x_t_outs[i].squeeze(0), steps=200)
                        out = []
                        for cont in self.conts:
                            out += cont.collections
                        return out

                    def frames(self) -> Iterator[Any]:
                        for i in range(len(x_t_outs)):
                            yield i, ax

                engine = Animator(trainset=self.trainset)
                interval = 1000 if strategy.__name__ == "class_stream" else 100
                fps = 1 if strategy.__name__ == "class_stream" else 30
                ani = FuncAnimation(
                    fig, engine.update, frames=engine.frames,
                    interval=interval, blit=True,
                    save_count=len(x_parts)
                )

                ax.legend(fontsize=24, markerscale=2.0)
                fig.tight_layout()
                writergif = animation.PillowWriter(fps=fps)
                ani.save(os.path.join(
                    i_outpath, f"animation-{filename}.gif"), writer=writergif)
                plt.clf()
                plt.cla()
                plt.close()
                # end the process of making an animation ===============

    def load_model(self, path: str) -> None:
        model_path = os.path.join(path, f"{self.args.model_hash}.pt")
        self.log(f"MODEL PATH: {model_path=}")
        if os.path.exists(model_path):
            saved = torch.load(model_path, map_location="cpu")
            self.epoch = saved["epoch"]
            self.best_nll = saved["best_nll"]

            self.model.load_state_dict(saved["state_dict"])
            self.model = self.model.to(self.args.device)
            self.optimizer.load_state_dict(saved["optimizer"])
            self.scheduler.load_state_dict(saved["scheduler"])

            self.finished = saved["finished"]
            self.tuned = saved.get("tuned", False)
            self.loaded = True

            s = f"loaded saved model: {self.epoch=} "
            s += f"{self.finished=}\nfrom path: {model_path}"
            self.log(s)

    def save_model(self, path: str, finished: bool = False) -> None:
        sd_path = os.path.join(path, f"{self.args.model_hash}.pt")
        save = dict(
            epoch=self.epoch,
            state_dict=self.model.state_dict(),
            optimizer=self.optimizer.state_dict(),
            scheduler=self.scheduler.state_dict(),
            finished=finished,
            best_nll=self.best_nll,
            tuned=self.tuned
        )
        torch.save(save, sd_path)

    def get_results_keys(
        self,
        additional_keys: Dict[str, Any] = {}
    ) -> Dict[str, Any]:

        hashable = isinstance(self.model, HashableModule)
        d = {"model": self.model.name}
        if hashable:
            d = self.model.get_results_keys()  # type: ignore

        return {
            **d,
            "epoch": self.epoch,
            "model_hash": self.args.model_hash,
            "run": self.args.run,
            "grad_set_size": self.args.grad_set_size,
            "grad_correction": self.args.grad_correction,
            "comment": self.args.comment,
        }

    def print_log_stats(
        self,
        prefix: str,
        names: List[str],
        values: List[Any]
    ) -> None:
        msg = f"({prefix}-{self.model.name} grad: {self.args.grad_set_size} "
        msg += f"train: {self.args.train_set_size}) epoch: "
        msg += f"{self.epoch}/{self.args.epochs} "
        for _, (n, v) in enumerate(zip(names, values)):
            msg += f"{n}: {v:.4f} "

        self.log(msg)

    def log_train_stats(self, path: str) -> Dict[str, float]:
        result_keys = {**self.get_results_keys(), "run_type": "train",
                       "train_set_size": self.args.train_set_size}
        file_prefix = "train"
        names, values = self.tr_stats.log_stats_df(
            os.path.join(path, f"{file_prefix}-results.csv"), result_keys)
        self.print_log_stats(self.args.mode, names, values)

        return {n: v for (n, v) in zip(names, values)}

    def log_test_stats(
        self,
        path: str,
        test_name: str = "test",
        additional_keys: Dict[str, Any] = {}
    ) -> Dict[str, float]:
        for i, ss in enumerate(self.args.test_set_sizes):
            results_keys = {
                **self.get_results_keys(),
                "run_type": test_name,
                "train_set_size": self.args.train_set_size,
                "test_set_size": ss,
                "ref_set_size": args.test_set_sizes[-1],
                **additional_keys,  # for adding extra keys for special tests
            }
            _, _ = self.full_te_stats[i].log_stats_df(
                os.path.join(path, f"{test_name}-results.csv"), results_keys)

            # overwrite the ref set size for this version which
            # contains smaller ref set
            results_keys = {**results_keys, "ref_set_size": ss}
            names, values = self.sub_te_stats[i].log_stats_df(
                os.path.join(path, f"{test_name}-results.csv"), results_keys)
            self.print_log_stats(f"{test_name}-{ss}", names, values)

        return {n: v for (n, v) in zip(names, values)}

    def log(self, msg: str) -> None:
        self.args.logger.info(msg)


if __name__ == "__main__":
    parser = ArgumentParser("argument parser for MBC MVN")

    parser.add_argument("--dataset", type=str,
                        default="toy-mixture-of-gaussians",
                        choices=["toy-mixture-of-gaussians"],
                        help="the dataset to use")
    parser.add_argument("--mvn-type", type=str, default="diag",
                        choices=["full", "diag"],
                        help="type of covariance for MoG dataset")
    parser.add_argument("--comment", type=str, default="",
                        help="comment to add to hash string and results file")
    parser.add_argument("--run", type=int, default=0, help="run number")
    parser.add_argument("--epoch-its", type=int, default=1000,
                        help="the number of iters to run in an epoch")
    parser.add_argument("--test-its", type=int, default=1000,
                        help="the number of iters to run in test mode")
    parser.add_argument("--val-its", type=int, default=200,
                        help="the number of iters to run in test mode")
    parser.add_argument("--epochs", type=int, default=50,
                        help="the number of epochs to run for")
    parser.add_argument("--batch-size", type=int, default=10,
                        help="batch size for training")
    parser.add_argument("--decay-epochs", type=int, nargs="+",
                        default=[35], help="the epochs to decay learning rate")
    parser.add_argument("--decay-gamma", type=float,
                        default=1e-1, help="the learning rate decay rate")
    parser.add_argument("--classes", type=int, default=4, help="classes")
    parser.add_argument("--train-set-size", type=int,
                        default=1024, help="set size for training")
    parser.add_argument("--h-dim", type=int, default=128, help="hidden dim")
    parser.add_argument("--test-set-sizes", type=int, nargs="+", default=[
                        8, 16, 32, 64, 128, 256, 512, 1024],
                        help="for testing effect of set size at test time")
    parser.add_argument("--test-set-size", type=int,
                        default=1024, help="full set size of testing")
    parser.add_argument("--pool", type=str, default="mean",
                        choices=["mean", "min", "max", "sum"],
                        help="pooling function if relevant")
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--k", type=int, default=4,
                        help="the number of slots to use in the sse encoder")
    parser.add_argument("--grad-set-size", type=int, default=1024,
                        help="the size of the partitioned gradient")
    parser.add_argument("--gpu", type=int, default=0, help="the gpu index")
    parser.add_argument("--heads", type=int, default=4,
                        help="number of attention heads")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="initial learning rate")

    choices = ["train", "test", "mbc-motivation-example"]
    parser.add_argument("--mode", type=str, choices=choices)
    parser.add_argument("--slot-type", type=str,
                        default="random", choices=["random", "deterministic"])
    parser.add_argument("--ln-slots", type=str2bool, default=True,
                        help="put layernorm pre-activation on the slots?")
    parser.add_argument("--grad-correction", type=str2bool, default=True,
                        help="whether or not to add a gradient correction")
    parser.add_argument("--ln-after", type=str2bool,
                        default=False, help="put layernorm after SSE")
    parser.add_argument("--slot-residual", type=str2bool, default=True,
                        help="put a residual connection on slots")
    parser.add_argument("--slot-drop", type=float, default=0.0,
                        help="slot dropout rate for the sse models")
    parser.add_argument("--model",
                        type=str,
                        choices=[
                            "deepsets", "sse",
                            "sse-umbc", "set-transformer",
                            "diff-em", "fspool",
                            "diff-em-umbc", "fspool-umbc",
                            "sse-hierarchical"
                        ]
                        )

    attn_acts = [
        "sigmoid", "softmax", "slot-sigmoid", "slot-softmax", "slot-exp"
    ]
    parser.add_argument("--attn-act", type=str,
                        choices=attn_acts,
                        default="the attention activation on MBC models")

    args = parser.parse_args()
    args.logger = set_logger("INFO")
    args.device = torch.device(f"cuda:{args.gpu}")
    args.dim = 2

    # seed before doing anything else with the dataloaders
    seed(args.run)

    train_ldr, _, _ = get_dataset(args)

    if args.mode == "mbc-motivation-example":
        x, y, pi, mu, sigma = train_ldr.dataset.sample(  # type: ignore
            args.batch_size * 10, args.test_set_size, args.classes)

    out_dim = 1 + 2 + (2 if args.mvn_type == "diag" else 4)
    in_dim = 2

    distr_emb_args = {}

    func_deref = {
        "deepsets": partial(
            MVNDeepSets, K=args.k, in_dim=in_dim, h_dim=args.h_dim,
            out_dim=out_dim, n_extractor_layers=1,
            n_decoder_layers=3, pool=args.pool
        ),
        "sse-umbc": partial(
            MVNSSEUMBC, in_dim=in_dim, h_dim=args.h_dim, out_dim=out_dim,
            n_extractor_layers=1, n_decoder_layers=3, K=args.k,
            h=args.h_dim, d=args.h_dim, d_hat=args.h_dim,
            heads=args.heads, slot_drop=args.slot_drop,
            attn_act=args.attn_act, slot_residual=args.slot_residual,
        ),
        "sse": partial(
            MVNSSE, in_dim=in_dim, h_dim=args.h_dim, out_dim=out_dim,
            n_extractor_layers=1, n_decoder_layers=3, K=args.k,
            h=args.h_dim, d=args.h_dim, d_hat=args.h_dim,
            heads=args.heads, slot_drop=args.slot_drop,
            attn_act=args.attn_act, slot_residual=args.slot_residual,
        ),
        "sse-hierarchical": partial(
            MVNSSEHierarchical, in_dim=in_dim, h_dim=args.h_dim, out_dim=out_dim,
            n_extractor_layers=1, n_decoder_layers=3, K=args.k,
            h=args.h_dim, d=args.h_dim, d_hat=args.h_dim,
            heads=args.heads, slot_drop=args.slot_drop,
            attn_act=args.attn_act, slot_residual=args.slot_residual,
        ),
        "set-transformer": partial(
            MVNSetTransformer, in_dim=in_dim, h_dim=args.h_dim,
            out_dim=out_dim, n_extractor_layers=1,
            n_decoder_layers=3, K=args.k, num_heads=args.heads,
        ),
        "diff-em": partial(
            MVNDiffEM, in_dim=in_dim, h_dim=args.h_dim,
            out_dim=out_dim, n_extractor_layers=1,
            n_decoder_layers=3, K=args.k,
            num_heads=5, num_proto=args.classes,
            num_ems=3, tau=1e-2,
        ),
        "fspool": partial(
            MVNFSPool, in_dim=in_dim, h_dim=args.h_dim,
            out_dim=out_dim, n_extractor_layers=1,
            n_decoder_layers=3, K=args.k,
        ),
        "diff-em-umbc": partial(
            MVNDiffEMUMBC, in_dim=in_dim, h_dim=args.h_dim,
            out_dim=out_dim, n_extractor_layers=1,
            n_decoder_layers=3, K=args.k,
            h=args.h_dim, d=args.h_dim, d_hat=args.h_dim,
            heads=args.heads, slot_drop=args.slot_drop,
            attn_act=args.attn_act, slot_residual=args.slot_residual,
            diffem_heads=5, num_proto=args.classes,
            num_ems=1, tau=1e-2,
        ),
        "fspool-umbc": partial(
            MVNFSPoolUMBC, in_dim=in_dim, h_dim=args.h_dim,
            out_dim=out_dim, n_extractor_layers=1,
            n_decoder_layers=3, K=args.k,
            h=args.h_dim, d=args.h_dim, d_hat=args.h_dim,
            heads=args.heads, slot_drop=args.slot_drop,
            attn_act=args.attn_act, slot_residual=args.slot_residual,
        )
    }
    model = func_deref[args.model]()  # type: ignore

    def make_args(model: Any) -> Namespace:
        views = f"GRAD SIZE: {args.grad_set_size}\n"
        views += f"GRAD CORRECTION: {args.grad_correction}"
        args.model_string = f"TRAIN SET SIZE: {args.train_set_size}\n"
        args.model_string += f"COMMENT: {args.comment}\nRUN: {args.run}\n"
        args.model_string += f"{views}\n\n" + str(model)
        args.model_hash = md5(args.model_string)
        return args

    def get_trainer(args: Namespace) -> Tuple[MVNTrainer, Namespace]:
        args = make_args(model)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        return MVNTrainer(args, model, optimizer, train_ldr.dataset), args

    if args.mode == "train":
        trainer, args = get_trainer(args)
        trainer.args.mode = "train"
        trainer.fit()
    elif args.mode == "test":
        tester, args = get_trainer(args)
        tester.load_model(tester.models_path)
        if not tester.finished:
            raise ValueError(
                "cannot test a model which has not finished training.")

        tester.test(its=1000)
        tester.log_test_stats(tester.results_path)
    elif args.mode == "mbc-motivation-example":
        trainer, args = get_trainer(args)
        trainer.load_model(trainer.models_path)
        with torch.no_grad():
            trainer.motivation_example(x, y, pi, mu, sigma)
    else:
        raise NotImplementedError()
