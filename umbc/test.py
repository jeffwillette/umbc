import copy
import os
import random
import unittest
from argparse import Namespace
from datetime import datetime
from functools import partial
from typing import Any, Callable, Dict, List, Tuple

import numpy as np
import pandas as pd  # type: ignore
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
from umbc.models.camelyon import (
    CamelyonABMIL,
    CamelyonDeepSets,
    CamelyonDSMIL,
    CamelyonFSPoolUMBC,
    CamelyonSSE,
    CamelyonSSEUMBC,
)
from umbc.models.diem.mog_models import EmbedderMoG
from umbc.models.layers.ab_mil import GatedAttention
from umbc.models.layers.bigset import (
    MBCExtractAndPool,
    MBCExtractor,
    ParallelMBCExtractAndPool,
)
from umbc.models.layers.deepsets import DeepSetsPooler
from umbc.models.layers.fspool import FSPool
from umbc.models.layers.resnet import ResNet50Pretrained
from umbc.models.layers.resnet_trunc import resnet50_trunc_baseline
from umbc.models.layers.sse import ParallelSSE, SlotSetEncoder
from umbc.models.mvn import (
    MVNSSE,
    MVNSSEUMBC,
    MVNDeepSets,
    MVNDiffEM,
    MVNDiffEMUMBC,
    MVNFSPool,
    MVNFSPoolUMBC,
    MVNSetTransformer,
    MVNSSEHierarchical,
)
from utils import get_module_root, md5, seed

T = torch.Tensor

GPU = 0


def check_close(A: T, B: T, atol: float = 1e-5) -> bool:
    result = torch.all(torch.isclose(A, B, rtol=0.0, atol=atol))
    if not result:
        print(f"failed (err: {torch.abs(A - B).amax()})")
    return result  # type: ignore


class SSETester(SlotSetEncoder):
    def check_minibatch_consistency(self, X: T, split_size: int) -> bool:
        # Sample Slots for Current S, encode the full set
        B, _, _, device = *X.size(), X.device
        S = self.sample_s()
        S_hat_X = self.forward(X=X, S=S)

        # Split X each with split_size elements, Encode splits.
        X = torch.split(X, split_size_or_sections=split_size, dim=1)
        S_hat_split = torch.zeros(B, self.slots.K, self.d_hat, device=device)
        C = torch.zeros(B, self.slots.K, self.heads, device=device)

        view_heads = (B, self.slots.K, self.heads, -1)
        view_std = (B, self.slots.K, -1)

        for split_i in X:
            S_out, S_hat_split_i, C_i = self.process_batch(X=split_i, S=S)

            S_hat_split += S_hat_split_i
            C += C_i

        S_hat_split = (S_hat_split.view(*view_heads) / C.view(*view_heads)).view(
            *view_std
        )

        if self.slot_residual:
            S_hat_split += S_out
        S_hat_split = self.norm_after(S_hat_split)

        return check_close(S_hat_X, S_hat_split)

    def check_input_invariance(self, X: T) -> bool:
        # Sample Slots for Current S, encode full set
        B, s, _ = X.size()
        S = self.sample_s()
        S_hat = self.forward(X=X, S=S)

        # Random permutations on X
        permutations = torch.randperm(s)
        X = X[:, permutations, :]

        S_hat_perm = self.forward(X=X, S=S)
        return check_close(S_hat, S_hat_perm)

    def check_slot_equivariance(self, X: T) -> bool:
        # Encode full set
        S = self.sample_s()
        S_hat = self.forward(X=X, S=S)

        # Random permutations on S
        permutations = torch.randperm(self.slots.K)
        S = S[:, permutations, :]
        S_hat_perm = self.forward(X=X, S=S)

        # Apply sampe permutation on S_hat
        S_hat = S_hat[:, permutations, :]
        return check_close(S_hat, S_hat_perm)


class TestDiem(unittest.TestCase):
    def test_smoketest_diem_mog(self) -> None:
        args = Namespace(
            out_type="select_best2",
            D=2,
            K=4,
            num_proto=4,
            num_ems=3,
            dim_feat=128,
            num_heads=5,
            tau=10.0,
        )
        distr_emb_args = {
            "dh": 128,
            "dout": 64,
            "num_eps": 5,
            "layers1": [128],
            "nonlin1": "relu",
            "layers2": [128],
            "nonlin2": "relu",
        }

        net = EmbedderMoG(
            args.D,
            args.K,
            out_type=args.out_type,
            num_proto=args.num_proto,
            num_ems=args.num_ems,
            dim_feat=args.dim_feat,
            num_heads=args.num_heads,
            tau=args.tau,
            distr_emb_args=distr_emb_args,
        )
        net = net.cuda()
        x = torch.randn(32, 100, 2).cuda()

        out = net(x)
        self.assertTrue(all([u == v for (u, v) in zip(out.size(), [32, 4, 5])]))


class TestFsPool(unittest.TestCase):
    def test_fspool(self) -> None:

        # first arg is channels (which is like a feature dimension)
        # second arg is pieces for piecewise linear functions
        #   which need to vary with set size. I don't fully understand
        #   what this is yet.

        pool = FSPool(2, 1)
        x = torch.arange(0, 2 * 3 * 4).view(3, 4, 2).float()  # (B, S, D)
        y = pool(x)
        self.assertTrue(all([a == b for (a, b) in zip(y.size(), [3, 2])]))


class TestParallelMBC(unittest.TestCase):
    @unittest.skip(
        "skipping embedding variance (this was only used to make a plot)"
    )  # noqa
    def test_embedding_vec_ll_umbc(self) -> None:
        root = get_module_root()
        dim = 128
        st_model = partial(
            MVNSetTransformer,
            in_dim=dim,
            h_dim=dim,
            out_dim=dim,
            n_extractor_layers=1,
            n_decoder_layers=1,
            K=1,
            num_heads=1,
        )

        umbc_model = partial(
            MVNSSEUMBC,
            in_dim=dim,
            h_dim=dim,
            out_dim=dim,
            n_extractor_layers=1,
            n_decoder_layers=1,
            K=1,
            h=dim,
            d=dim,
            d_hat=dim,
            heads=1,
            attn_act="sigmoid",
        )

        fspool_model = partial(
            MVNFSPool,
            in_dim=dim,
            h_dim=dim,
            out_dim=dim,
            n_extractor_layers=1,
            n_decoder_layers=1,
            K=1,
        )

        diffem_model = partial(
            MVNDiffEM,
            in_dim=dim,
            h_dim=dim,
            out_dim=dim,
            n_extractor_layers=1,
            n_decoder_layers=1,
            K=1,
            num_heads=1,
        )

        st = st_model()
        umbc_st = umbc_model()
        fspool = fspool_model()
        diffem = diffem_model()

        embeddings = []
        data = {"chunk_size": [], "universal": [], "index": [], "name": []}

        n = 256
        x = torch.cat(
            [
                torch.randn(n, dim),
                torch.rand(n, dim) * 6 - 3,
                torch.zeros(n, dim).exponential_(),
                torch.zeros(n, dim).cauchy_(),
            ]
        ).cuda(GPU)

        st, umbc_st, fspool, diffem = (
            st.cuda(GPU),
            umbc_st.cuda(GPU),
            fspool.cuda(GPU),
            diffem.cuda(GPU),
        )

        umbc_st.pre_forward_mbc()
        st.eval()
        umbc_st.eval()
        fspool.eval()
        diffem.eval()

        with torch.no_grad():
            for i in range(100):  # 100 random samples of tensor
                print(i)
                for chunks in [1, 2, 4, 8, 16, 32]:  # 6 chunk sizes
                    x_perm = x[torch.randperm(x.size(0))]
                    x_chunked = x_perm.chunk(chunks, dim=0)

                    models = [st, umbc_st, fspool, diffem]
                    names = ["st", "umbc_st", "fspool", "diffem"]
                    universal = [False, True, False, False]
                    for mdl, name, is_universal in zip(models, names, universal):

                        x_t, c_t = None, None
                        for chnk in x_chunked:
                            x_t, c_t = mdl.forward_mbc(  # type: ignore
                                chnk.unsqueeze(0), X_prev=x_t, c_prev=c_t, grad=False
                            )

                        out = (
                            mdl.post_forward_mbc(x_t, c=c_t)  # type: ignore
                            .cpu()
                            .view(-1)
                            .numpy()
                        )
                        embeddings.append(out)
                        data["chunk_size"].append(chunks)
                        data["universal"].append(is_universal)
                        data["name"].append(name)
                        data["index"].append(i)
                        print(".", end="", flush=True)

        df = pd.DataFrame(data)
        df.to_csv(os.path.join(root, "umbc", "plots", "st-umbcst-embeddings.csv"))

        # pickle, do not stack
        np.save(
            os.path.join(root, "umbc", "plots", "st-umbcst-embeddings.npy"), embeddings
        )

    @unittest.skip("skipping memory usage (this was only used to make a plot)")
    def test_mvn_memory_usage(self) -> None:
        dim = 128

        st = partial(
            MVNSetTransformer,
            in_dim=2,
            h_dim=dim,
            out_dim=5,
            n_extractor_layers=1,
            n_decoder_layers=3,
            K=4,
            num_heads=4,
        )

        diem = partial(
            MVNDiffEM,
            in_dim=2,
            h_dim=dim,
            out_dim=5,
            n_extractor_layers=1,
            n_decoder_layers=3,
            K=4,
            num_heads=5,
        )

        fspool = partial(
            MVNFSPool,
            in_dim=2,
            h_dim=dim,
            out_dim=5,
            n_extractor_layers=1,
            n_decoder_layers=3,
            K=4,
        )

        umbc = partial(
            MVNSSEUMBC,
            in_dim=2,
            h_dim=dim,
            out_dim=5,
            n_extractor_layers=1,
            n_decoder_layers=3,
            K=4,
            h=dim,
            d=dim,
            d_hat=dim,
            heads=4,
            slot_drop=0.0,
            attn_act="softmax",
            slot_residual=True,
        )

        fspool_umbc = partial(
            MVNFSPoolUMBC,
            in_dim=2,
            h_dim=dim,
            out_dim=5,
            n_extractor_layers=1,
            n_decoder_layers=3,
            K=4,
            h=dim,
            d=dim,
            d_hat=dim,
            heads=4,
            slot_drop=0.0,
            attn_act="softmax",
            slot_residual=True,
        )

        diem_umbc = partial(
            MVNDiffEMUMBC,
            in_dim=2,
            h_dim=dim,
            out_dim=5,
            n_extractor_layers=1,
            n_decoder_layers=3,
            K=4,
            h=dim,
            d=dim,
            d_hat=dim,
            heads=4,
            slot_drop=0.0,
            attn_act="softmax",
            slot_residual=True,
            diffem_heads=5,
        )

        ds = partial(
            MVNDeepSets,
            K=4,
            in_dim=2,
            h_dim=dim,
            out_dim=5,
            n_extractor_layers=1,
            n_decoder_layers=3,
            pool="max",
        )

        sse = partial(
            MVNSSE,
            in_dim=2,
            h_dim=dim,
            out_dim=5,
            n_extractor_layers=1,
            n_decoder_layers=3,
            K=4,
            h=dim,
            d=dim,
            d_hat=dim,
            heads=4,
            slot_drop=0.0,
            attn_act="slot-sigmoid",
            slot_residual=True,
        )

        sse_hierarchical = partial(
            MVNSSEHierarchical,
            in_dim=2,
            h_dim=dim,
            out_dim=5,
            n_extractor_layers=1,
            n_decoder_layers=3,
            K=4,
            h=dim,
            d=dim,
            d_hat=dim,
            heads=4,
            slot_drop=0.0,
            attn_act="slot-sigmoid",
            slot_residual=True,
        )

        for mdl, name in zip(
            [st, umbc, ds, sse, sse_hierarchical, diem, fspool, diem_umbc, fspool_umbc],
            [
                "st",
                "umbc",
                "ds",
                "sse",
                "sse-hierarchical",
                "diem",
                "fspool",
                "diem_umbc",
                "fspool_umbc",
            ],
        ):
            mem_used = []  # type: ignore
            # 6 chunk sizes

            if name != "sse-hierarchical":
                continue

            for gc_size in [8, 16, 32, 64, 128, 256, 512]:
                model = mdl().cuda(GPU)  # type: ignore
                loss_fn = nn.MSELoss()

                d: Any = {}

                def get_hook(n: str) -> Any:
                    def hook(module: nn.Module, inp: T, outp: T) -> None:
                        if torch.is_grad_enabled():

                            if len(inp) > 0:
                                inp = torch.cat([v.reshape(-1) for v in inp])
                                inp_hash = md5(str(inp.data.tolist()))

                                if inp_hash not in d.keys():
                                    d[inp_hash] = inp[0]

                            if isinstance(outp, T):
                                outp_hash = md5(str(outp.data.tolist()))
                                if outp_hash not in d.keys():
                                    d[outp_hash] = outp
                            elif isinstance(outp, Tuple):
                                outp = torch.cat([v.reshape(-1) for v in outp])
                                outp_hash = md5(str(outp.data.tolist()))
                                if outp_hash not in d.keys():
                                    d[outp_hash] = outp

                    return hook

                for n, m in model.named_modules():
                    m.register_forward_hook(get_hook(n))
                    m.hook = get_hook(n)

                loss_fn.register_forward_hook(get_hook("loss"))

                grad_chunk = torch.randn(1, gc_size, 2).cuda(GPU)

                if name in ["st", "diem", "fspool"]:
                    grad_chunk = torch.randn(1, gc_size, 2).cuda(GPU)
                    out = model(grad_chunk)
                else:
                    grad_chunk = torch.randn(1, 8, 2).cuda(GPU)
                    model.pre_forward_mbc()
                    x_t, c_t = model.forward_mbc(grad_chunk, grad=True)

                    if gc_size > 8:
                        with torch.no_grad():
                            no_grad_chunk = torch.randn(1, gc_size - 8, 2).cuda(GPU)
                            x_t, c_t = model.forward_mbc(  # type: ignore
                                no_grad_chunk, X_prev=x_t, c_prev=c_t, grad=False
                            )

                    out = model.post_forward_mbc(x_t, c=c_t)  # type: ignore
                    out = model.decoder(out)

                loss = loss_fn(out, torch.zeros_like(out))

                total = 0
                for _, v in d.items():
                    total += 4 * v.numel()

                mem_used.append(total / 1000)

                for item in [model, grad_chunk, loss, out]:
                    del item

            print(f"{name}: {mem_used}")

        N = 10000
        lst: Any = []
        lst.extend([(st, "st") for _ in range(N)])
        lst.extend([(umbc, "umbc") for _ in range(N)])
        lst.extend([(ds, "ds") for _ in range(N)])
        lst.extend([(sse, "sse") for _ in range(N)])
        lst.extend([(sse, "sse-hierarchical") for _ in range(N)])
        lst.extend([(diem, "diem") for _ in range(N)])
        lst.extend([(fspool, "fspool") for _ in range(N)])
        lst.extend([(diem_umbc, "diem_umbc") for _ in range(N)])
        lst.extend([(fspool_umbc, "fspool_umbc") for _ in range(N)])
        random.shuffle(lst)

        times: Dict[str, Any] = {
            "st": [],
            "umbc": [],
            "ds": [],
            "sse": [],
            "sse-hierarchical": [],
            "diem": [],
            "fspool": [],
            "diem_umbc": [],
            "fspool_umbc": [],
        }
        for i, (mdl, name) in enumerate(lst):
            print(i, end=" ")
            model = mdl().cuda(GPU)  # type: ignore
            model.eval()

            start = datetime.now()

            model.pre_forward_mbc()

            inputs = torch.randn(1, 8, 2).cuda(GPU)
            x_t, c_t = None, None
            for chnk in inputs.split(8, dim=1):
                x_t, c_t = model.forward_mbc(
                    grad_chunk, X_prev=x_t, c_prev=c_t, grad=False
                )

            out = model.post_forward_mbc(x_t, c=c_t)
            out = model.decoder(out)

            times[name].append((datetime.now() - start).total_seconds())

        for k in times.keys():
            t = np.array(times[k])
            print(f"{k} {t.mean()=} {t.std()=}")

    def test_parallel_sse_hash_collisions(self) -> None:
        hashes = []
        for n_parallel in [1, 2]:
            model = ParallelSSE(
                K=1, h=4, d=1, d_hat=4, n_parallel=n_parallel
            )  # type: ignore
            hashes.append(md5(str(model)))

        self.assertTrue(len(hashes) == len(list(set(hashes))))
        self.assertTrue(
            all([u == v for (u, v) in zip(sorted(hashes), sorted(list(set(hashes))))])
        )  # noqa

    def test_sse_hash_collisions(self) -> None:
        hashes = []
        for slot_type in ["random", "deterministic"]:
            for heads in [1, 2]:
                for K in [1, 2]:
                    for d in [1, 2]:
                        for h in [2, 4]:
                            for ln_slots in [True, False]:
                                for slot_drop in [0.0, 0.5]:
                                    for attn_act in ["sigmoid", "softmax"]:
                                        for slot_residual in [False, True]:
                                            model = SlotSetEncoder(
                                                K=K,
                                                h=h,
                                                d=d,
                                                d_hat=h,
                                                slot_type=slot_type,
                                                ln_slots=ln_slots,
                                                heads=heads,
                                                slot_drop=slot_drop,
                                                attn_act=attn_act,
                                                slot_residual=slot_residual,
                                            )
                                            hashes.append(md5(str(model)))
                                            print(".", end="", flush=True)

        self.assertTrue(len(hashes) == len(list(set(hashes))))
        self.assertTrue(
            all([u == v for (u, v) in zip(sorted(hashes), sorted(list(set(hashes))))])
        )  # noqa

    def test_parallel_sse(self) -> None:
        for attn_act in [
            "softmax",
            "sigmoid",
            "slot-sigmoid",
            "slot-softmax",
            "slot-exp",
        ]:
            for n_parallel in [1, 2]:
                model = ParallelSSE(
                    K=16,
                    h=3,
                    d=3,
                    d_hat=16,
                    heads=4,
                    slot_type="deterministic",
                    n_parallel=n_parallel,
                    attn_act=attn_act,
                )
                model.eval()  # make sure dropout is off

                x = torch.randn(32, 200, 3)
                full = model(x)

                model.pre_forward_mbc()
                x_prev, c_prev = None, None
                for s in x.split(50, dim=1):
                    x_prev, c_prev = model.forward_mbc(s, X_prev=x_prev, c_prev=c_prev)

                assert x_prev is not None
                batched = model.post_forward_mbc(x_prev, c_prev)

                passed = check_close(full, batched)
                if not passed:
                    print(f"max err: {torch.abs(full - batched).amax()=}")
                self.assertTrue(passed)
                print(".", end="", flush=True)

    def test_mbc_sse_partitioned_forward(self) -> None:
        for slot_type in ["random", "deterministic"]:
            for n_heads in [1, 4]:
                for attn_act in [
                    "sigmoid",
                    "slot-sigmoid",
                    "slot-softmax",
                    "slot-exp",
                    "softmax",
                ]:
                    for slot_residual in [False, True]:
                        sse = SlotSetEncoder(
                            K=16,
                            h=3,
                            d=3,
                            d_hat=16,
                            heads=4,
                            slot_type="deterministic",
                            attn_act=attn_act,
                        )

                        x = torch.randn(32, 200, 3)

                        batched = sse.partitioned_forward(x[:, :100], x[:, 100:])
                        full = sse(x)
                        passed = check_close(full, batched)

                        if not passed:
                            print(
                                f"max err: {torch.abs(full - batched).amax()=}"
                            )  # noqa
                        self.assertTrue(passed)
                        print(".", end="", flush=True)

                        do_profile = False
                        if do_profile:
                            x = torch.randn(32, 1000, 3).cuda(GPU)
                            sse.zero_grad()
                            sse = sse.cuda(GPU)

                            batched = sse.partitioned_forward(x[:, :100], x[:, 100:])
                            partial = torch.cuda.memory_allocated(GPU) / 1000000

                            # detached_batch = torch.clone(batched.detach())
                            del batched
                            sse.zero_grad()

                            full = sse(x)
                            full = torch.cuda.memory_allocated(GPU) / 1000000
                            print(f"{partial=} {full=}")

                            del full

    def test_parallel_sse_partitioned_forward(self) -> None:
        for attn_act in ["slot-sigmoid"]:
            for n_parallel in [2]:
                umbc = ParallelSSE(
                    K=16,
                    h=3,
                    d=3,
                    d_hat=16,
                    heads=4,
                    slot_type="deterministic",
                    n_parallel=n_parallel,
                    attn_act=attn_act,
                )

                x = torch.randn(32, 10000, 3)

                out = umbc(x)
                loss = (out**2).sum()
                loss.backward()

                umbc.grad_correct(100)

                umbc.zero_grad()
                batched = umbc.partitioned_forward(x[:, :100], x[:, 100:])

                umbc.zero_grad()
                full = umbc(x)

                passed = check_close(full, batched)
                if not passed:
                    print(f"max err: {torch.abs(full - batched).amax()=}")
                self.assertTrue(passed)
                print(".", end="", flush=True)

    def test_mbc_slot_set_encoder(self) -> None:
        # THESE ARE FROM THE MBC FILE
        B = 256  # Batch Size
        n = 50  # Set Size
        d = 3  # Element Dimension
        K = 8  # Number of Slots
        h = 3  # Slot Size
        d_hat = 8  # Linear Projection Dimension
        split_size = 10
        X = torch.rand(B, n, d)

        # for slot_type in ["random", "deterministic"]:
        for slot_type in ["deterministic"]:
            for n_heads in [1, 4]:
                for attn_act in [
                    "sigmoid",
                    "slot-sigmoid",
                    "slot-softmax",
                    "slot-exp",
                    "softmax",
                ]:
                    for slot_residual in [False, True]:
                        slot_encoder = SSETester(
                            K=K,
                            h=h,
                            d=d,
                            d_hat=d_hat,
                            slot_type=slot_type,
                            heads=n_heads,
                            attn_act=attn_act,
                            slot_residual=slot_residual,
                        )
                        slot_encoder.eval()

                        one = slot_encoder.check_minibatch_consistency(
                            X, split_size=split_size
                        )
                        two = slot_encoder.check_input_invariance(X)
                        three = slot_encoder.check_slot_equivariance(X)
                        self.assertTrue(all([two, three]))
                        self.assertTrue(all([one, three]))
                        self.assertTrue(all([one, two]))
                        self.assertTrue(all([one, two, three]))
                        print("...", end="", flush=True)

    def setup_bigset_extractor(self) -> Tuple[List[Any], ...]:
        K = 8
        h = 4
        d = 4
        poolers = [
            partial(DeepSetsPooler, pool="max", h_dim=4),
            partial(DeepSetsPooler, pool="min", h_dim=4),
            partial(DeepSetsPooler, pool="mean", h_dim=4),
            partial(DeepSetsPooler, pool="sum", h_dim=4),
            partial(
                SlotSetEncoder,
                K=K,
                h=h,
                d=d,
                d_hat=8,
                slot_type="deterministic",
                slot_residual=False,
                ln_after=True,
            ),
            partial(
                SlotSetEncoder,
                K=K,
                h=h,
                d=d,
                d_hat=8,
                slot_type="deterministic",
                slot_residual=False,
                ln_after=False,
            ),
            partial(
                SlotSetEncoder,
                K=K,
                h=h,
                d=d,
                d_hat=8,
                slot_type="deterministic",
                slot_residual=True,
                ln_after=False,
            ),
            partial(
                SlotSetEncoder,
                K=K,
                h=h,
                d=d,
                d_hat=8,
                slot_type="deterministic",
                slot_residual=True,
                ln_after=True,
            ),
            partial(
                SlotSetEncoder,
                K=K,
                h=h,
                d=d,
                d_hat=8,
                slot_type="deterministic",
                slot_residual=True,
                ln_after=True,
            ),
            partial(
                SlotSetEncoder,
                K=K,
                h=h,
                d=d,
                d_hat=8,
                slot_type="deterministic",
                slot_residual=True,
                heads=1,
            ),
            partial(SlotSetEncoder, K=K, h=h, d=d, d_hat=8, slot_type="random"),
            partial(
                SlotSetEncoder,
                K=K,
                h=h,
                d=d,
                d_hat=8,
                slot_type="deterministic",
                slot_residual=False,
                attn_act="softmax",
            ),
            partial(
                SlotSetEncoder,
                K=K,
                h=h,
                d=d,
                d_hat=8,
                slot_type="deterministic",
                slot_residual=False,
                attn_act="sigmoid",
            ),
            partial(
                SlotSetEncoder,
                K=K,
                h=h,
                d=d,
                d_hat=8,
                slot_type="deterministic",
                slot_residual=False,
                attn_act="slot-sigmoid",
            ),
            partial(
                SlotSetEncoder,
                K=K,
                h=h,
                d=d,
                d_hat=8,
                slot_type="deterministic",
                slot_residual=False,
                attn_act="slot-softmax",
            ),
            partial(
                SlotSetEncoder,
                K=K,
                h=h,
                d=d,
                d_hat=8,
                slot_type="deterministic",
                slot_residual=False,
                attn_act="slot-exp",
            ),
            partial(
                ParallelSSE,
                K=K // 4,
                h=h,
                d=d,
                d_hat=8,
                slot_type="deterministic",
                n_parallel=1,
            ),
            partial(
                ParallelSSE,
                K=K // 4,
                h=h,
                d=d,
                d_hat=8,
                slot_type="deterministic",
                n_parallel=4,
            ),
            partial(
                ParallelSSE,
                K=K // 4,
                h=h,
                d=d,
                d_hat=8,
                slot_type="random",
                n_parallel=4,
            ),
        ]

        wrapper_classes = [
            MBCExtractAndPool,
            MBCExtractAndPool,
            MBCExtractAndPool,
            MBCExtractAndPool,
            MBCExtractAndPool,
            MBCExtractAndPool,
            MBCExtractAndPool,
            MBCExtractAndPool,
            MBCExtractAndPool,
            MBCExtractAndPool,
            MBCExtractAndPool,
            MBCExtractAndPool,
            MBCExtractAndPool,
            MBCExtractAndPool,
            MBCExtractAndPool,
            MBCExtractAndPool,
            ParallelMBCExtractAndPool,
            ParallelMBCExtractAndPool,
            ParallelMBCExtractAndPool,
        ]

        extractors = [
            partial(
                MBCExtractor,
                n_layers=2,
                in_dim=4,
                h_dim=4,
                out_dim=4,
                activation="tanh",
            ),
            partial(
                MBCExtractor,
                n_layers=3,
                in_dim=4,
                h_dim=4,
                out_dim=4,
                activation="relu",
            ),
        ]
        return wrapper_classes, poolers, extractors

    def test_umbc_func_part_forw_mbc_mask(self) -> None:
        wrapper_classes, poolers, extractors = self.setup_bigset_extractor()
        for wrapper_class, pooler in zip(wrapper_classes, poolers):
            for extractor in extractors:
                bigset_func = wrapper_class(extractor(), pooler())

                x_list, x_padded_list, mask_list = [], [], []
                instance_length = np.random.randint(1, 1000, (32,))
                for n in instance_length:
                    x = torch.randn((n, 4))
                    pad = torch.zeros((1000 - n, 4))
                    mask = torch.ones(1000)
                    mask[n:] = 0.0

                    x_list.append(x)
                    x_padded_list.append(torch.cat((x, pad), dim=0))
                    mask_list.append(mask)

                # call the setup function once to sample slots
                bigset_func.pre_forward_mbc()

                full = torch.cat([bigset_func(v.unsqueeze(0)) for v in x_list])

                padded_input = torch.stack(x_padded_list)
                mask = torch.stack(mask_list)

                x_prev, c_prev = None, None
                for i, (x_chnk, mask_chnk) in enumerate(
                    zip(padded_input.chunk(100, dim=1), mask.chunk(100, dim=1))
                ):
                    x_prev, c_prev = bigset_func.forward_mbc(
                        X=x_chnk,
                        X_prev=x_prev,
                        c_prev=c_prev,
                        grad=i == 0,
                        mask=mask_chnk,
                    )

                assert x_prev is not None
                x_part = bigset_func.post_forward_mbc(x_prev, c_prev, mask=mask)

                passed = check_close(full, x_part)
                if not passed:
                    print(f"max err: {torch.abs(full - x_part).amax()=}")
                self.assertTrue(passed, bigset_func.mbc_pooler)
                print(".", end="", flush=True)

    def test_umbc_func_forward_mbc(self) -> None:
        wrapper_classes, poolers, extractors = self.setup_bigset_extractor()
        for wrapper_class, pooler in zip(wrapper_classes, poolers):
            for extractor in extractors:
                bigset_func = wrapper_class(extractor(), pooler()).cuda(GPU)

                bigset_func.zero_grad()

                x = torch.randn(32, 1000, 4).cuda(GPU)
                rand_seed = np.random.randint(0, 1000)

                seed(rand_seed)
                bigset_func.pre_forward_mbc()
                x_prev, c_prev = None, None
                for i, chunk in enumerate(x.chunk(100, dim=1)):
                    x_prev, c_prev = bigset_func.forward_mbc(
                        X=chunk, X_prev=x_prev, c_prev=c_prev, grad=i == 0
                    )

                assert x_prev is not None
                x_part = bigset_func.post_forward_mbc(x_prev, c_prev)

                mem_partial = torch.cuda.memory_allocated(GPU) / 1000000

                x_part = x_part.detach()
                del x_prev
                del c_prev
                bigset_func.zero_grad()

                seed(rand_seed)
                bigset_func.zero_grad()
                full = bigset_func(x)
                mem_full = torch.cuda.memory_allocated(GPU) / 1000000

                print_profile = False
                if print_profile:
                    print(f"{mem_partial=} {mem_full=}")

                passed = check_close(full, x_part)
                if not passed:
                    print(f"max err: {torch.abs(full - x_part).amax()=}")
                self.assertTrue(passed)
                print(".", end="", flush=True)

                del x_part
                del full

    def test_umbc_func_partitioned_forward(self) -> None:
        wrapper_classes, poolers, extractors = self.setup_bigset_extractor()
        for wrapper_class, pooler in zip(wrapper_classes, poolers):
            for extractor in extractors:
                bigset_func = wrapper_class(extractor(), pooler())

                x = torch.randn(32, 100, 4)
                rand_seed = np.random.randint(0, 1000)

                seed(rand_seed)
                full = bigset_func(x)
                seed(rand_seed)
                part = bigset_func.partitioned_forward(x[:, :10], x[:, 10:])

                passed = check_close(full, part)
                if not passed:
                    print(f"max err: {torch.abs(full - part).amax()=}")
                self.assertTrue(passed)
                print(".", end="", flush=True)

    def test_umbc_func_grad_correct(self) -> None:
        wrapper_classes, poolers, extractors = self.setup_bigset_extractor()
        for wrapper_class, pooler in zip(wrapper_classes, poolers):
            for extractor in extractors:
                bigset_func = wrapper_class(extractor(), pooler())

                x = torch.randn(32, 100, 4)
                rand_seed = np.random.randint(0, 1000)

                full_model = copy.deepcopy(bigset_func)
                part_model = copy.deepcopy(bigset_func)
                part_model_two = copy.deepcopy(bigset_func)

                seed(rand_seed)
                full = full_model(x)
                full_loss = (full**2).sum()
                full_loss.backward()

                seed(rand_seed)
                part = part_model.partitioned_forward(x[:, :50], x[:, 50:])
                part_loss = (part**2).sum()
                part_loss.backward()

                seed(rand_seed)
                part_two = part_model_two.partitioned_forward(x[:, :50], x[:, 50:])
                part_two_loss = (part_two**2).sum()
                part_two_loss.backward()
                part_model_two.grad_correct(2)

                # ignore the layernorm after the deepsets pooling
                not_close, close = False, True

                for ((n1, p1), (n2, p2), (n3, p3)) in zip(
                    full_model.named_parameters(),
                    part_model.named_parameters(),
                    part_model_two.named_parameters(),
                ):
                    if "norm_after" not in n1:
                        # these layers are small, and sometimes a bias layer
                        # might have an equal grad so we require only one
                        # parameter not be equal
                        p1p2_close = torch.isclose(
                            torch.abs(p1.grad - p2.grad).sum(), torch.tensor(0.0)
                        )
                        if not p1p2_close:
                            not_close = True

                        # this case is stricter and we require all parameters
                        # to be equal
                        p2p3_close = check_close(p2.grad * 2, p3.grad)
                        if not p2p3_close:
                            close = False

                self.assertTrue(not_close, bigset_func.mbc_pooler)
                print(".", end="", flush=True)
                self.assertTrue(close, bigset_func.mbc_pooler)
                print(".", end="", flush=True)

    def test_smoketest_mvn_models(self) -> None:
        mvn_functions = [
            MVNDeepSets(
                K=4,
                in_dim=2,
                h_dim=4,
                out_dim=5,
                n_extractor_layers=2,
                n_decoder_layers=2,
            ),
            MVNSSE(
                in_dim=2,
                h_dim=4,
                out_dim=5,
                n_extractor_layers=2,
                n_decoder_layers=2,
                K=4,
                h=3,
                d=4,
                d_hat=4,
                heads=1,
            ),
            MVNSSEUMBC(
                in_dim=2,
                h_dim=4,
                out_dim=5,
                n_extractor_layers=2,
                n_decoder_layers=2,
                K=4,
                h=3,
                d=4,
                d_hat=4,
                heads=1,
            ),
            MVNSetTransformer(
                in_dim=2,
                h_dim=4,
                out_dim=5,
                n_extractor_layers=2,
                n_decoder_layers=2,
                K=4,
                num_heads=1,
            ),
            MVNDiffEM(
                in_dim=2,
                h_dim=4,
                out_dim=5,
                n_extractor_layers=2,
                n_decoder_layers=2,
                K=4,
                num_heads=5,
            ),
            MVNFSPool(
                in_dim=2,
                h_dim=4,
                out_dim=5,
                n_extractor_layers=2,
                n_decoder_layers=2,
                K=4,
            ),
            MVNFSPoolUMBC(
                in_dim=2,
                h_dim=4,
                out_dim=5,
                n_extractor_layers=2,
                n_decoder_layers=2,
                K=4,
                h=3,
                d=4,
                d_hat=4,
                heads=1,
            ),
            MVNDiffEMUMBC(
                in_dim=2,
                h_dim=4,
                out_dim=5,
                n_extractor_layers=2,
                n_decoder_layers=2,
                K=4,
                h=3,
                d=4,
                d_hat=4,
                heads=1,
                diffem_heads=5,
                tau=0.01,
            ),
        ]

        for function in mvn_functions:
            x = torch.randn(32, 100, 2)
            out = function(x)
            b, s, d = out.size()

            for real, expected in zip((b, s, d), (32, 4, 5)):
                self.assertEqual(real, expected)
                print(".", end="", flush=True)

    def test_camelyon_non_mbc_models(self) -> None:
        if not os.path.exists(os.path.join(get_module_root(), ".pretrained")):
            print("need to download pretrained resnet to run this test")
            return

        h_dim = 8
        models = [CamelyonDSMIL(1, h_dim), CamelyonABMIL(1, h_dim)]

        for mdl in models:
            x = torch.randn(1, 100, 3, 256, 256)
            x_ft = mdl.extractor(x)  # type: ignore

            sizes = [x_ft.size(0) == 1, x_ft.size(1) == 100, x_ft.size(2) == 256]
            self.assertTrue(all(sizes), sizes)
            print(".", end="", flush=True)

            inst, bag = mdl.decoder(x_ft)  # type: ignore
            sizes = [
                inst.size(0) == 100,
                inst.size(1) == 1,
                bag.size(0) == 1,
                bag.size(1) == 1,
            ]
            self.assertTrue(all(sizes), sizes)
            print(".", end="", flush=True)

    def test_camelyon_mbc_models(self) -> None:
        if not os.path.exists(os.path.join(get_module_root(), ".pretrained")):
            print("need to download pretrained resnet to run this test")
            return

        h_dim = 8
        models = [
            CamelyonDeepSets(out_dim=1, n_decoder_layers=2, h_dim=h_dim),
            CamelyonSSE(h=h_dim, d=256, d_hat=h_dim, out_dim=1, n_decoder_layers=2),
            CamelyonSSEUMBC(
                K=4, h=h_dim, d=256, d_hat=h_dim, out_dim=1, n_decoder_layers=2
            ),
            CamelyonFSPoolUMBC(
                K=4, h=h_dim, d=256, d_hat=h_dim, out_dim=1, n_decoder_layers=2
            ),
        ]

        for mdl in models:
            mdl.eval()
            x = torch.randn(1, 100, 3, 256, 256)
            mdl.pre_forward_mbc()
            chnk_ft, x_t, c_t = [], None, None
            for chnk in x.chunk(10, dim=1):
                x_ft, x_t, c_t = mdl.forward_mbc(  # type: ignore
                    chnk, X_prev=x_t, c_prev=c_t, grad=False
                )

                chnk_ft.append(x_ft)  # type: ignore

            x_out = mdl.post_forward_mbc(x_t, c=c_t)  # type: ignore
            mbc_inst, mbc_bag = mdl.decoder(torch.cat(chnk_ft, dim=1), x_out)  # type: ignore  # noqa

            x = mdl.extractor(x)  # type: ignore
            p_x = mdl.mbc_pooler(x)  # type: ignore

            inst, bag = mdl.decoder(x, p_x)  # type: ignore
            self.assertTrue(check_close(inst, mbc_inst))
            self.assertTrue(check_close(bag, mbc_bag))
            print(".", end="", flush=True)

    def test_resnet_imgnet_extractor(self) -> None:
        rn = ResNet50Pretrained()
        out = rn(torch.randn(1, 3, 256, 256))
        assert out is not None
        # print(out.size())

    def test_resnet_imgnet_trunc_extractor(self) -> None:
        rn = resnet50_trunc_baseline(pretrained=True)
        out = rn(torch.randn(1, 3, 256, 256))
        assert out is not None
        # print(out.size())

    def test_gated_attention(self) -> None:
        L, D, K = 1024, 512, 1
        attn = GatedAttention(L, D, K)
        out = attn(torch.randn(234, L))
        assert out is not None
        # print(out.size())


def setup(rank: int, world_size: int) -> None:
    """
    Setup code comes directly from the docs:
    https://pytorch.org/tutorials/intermediate/ddp_tutorial.html
    """
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29500"

    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.manual_seed(42)
    torch.cuda.set_device(rank)


def cleanup() -> None:
    dist.destroy_process_group()


def post_avg_closure(prefix: str) -> Callable[[T], None]:
    def post_average(g: T) -> None:
        print(f"Post-DDP hook {prefix} ({g.device}): {g.view(-1)[0]}")

    return post_average


def worker(rank: int, world_size: int) -> None:
    # Set up multiprocessing stuff
    setup(rank, world_size)
    model = CamelyonDeepSets(out_dim=1, n_decoder_layers=1).to(rank)

    for set_size in [32, 5]:
        seed(1)
        data = [torch.randn(32, 3, 256, 256) for _ in range(4)]

        # Create a trivial model
        remove_ext_handles = model.extractor.register_grad_correct_hooks(
            grad_size=5, set_size=set_size
        )
        remove_pooler_handles = model.mbc_pooler.register_grad_correct_hooks(
            grad_size=5, set_size=set_size
        )

        # model.weight.register_hook(pre_average)
        ddp_model = DDP(model, device_ids=[rank])
        # for n, p in ddp_model.module.extractor.named_parameters():
        #     p.register_hook(post_avg_closure("extractor"))

        # for n, p in ddp_model.module.mbc_pooler.named_parameters():
        #     p.register_hook(post_avg_closure("pooler"))

        # for n, p in ddp_model.module.decoder.named_parameters():
        #     p.register_hook(post_avg_closure("decoder"))

        # Backprop!
        inst_out, bag_out = ddp_model(data, grad_size=8)
        loss = inst_out.pow(2).mean() + bag_out.pow(2).mean()
        loss = loss / 2
        loss.backward()

        # Check what's left in the gradient tensors
        for n, p in ddp_model.named_parameters():
            if "norm_after" in n:
                continue

            if p.grad is not None:
                # print(n, p.grad.view(-1)[0])
                continue

        remove_ext_handles()
        remove_pooler_handles()

        ddp_model.zero_grad()

        import time

        time.sleep(1)
    cleanup()


class TestDDPGradCorrect(unittest.TestCase):
    def test_grad_correct_with_hooks(self) -> None:
        if torch.cuda.device_count() < 2:
            print("need more than two cuda devices for DDP testing")
            return

        world_size = 2
        mp.spawn(worker, args=(world_size,), nprocs=world_size, join=True)
