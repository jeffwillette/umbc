
from functools import partial
from typing import Any, Tuple
import argparse
import torch
from torch import nn

from umbc.models.cnp import MBC, UMBC, DeepSet, SetTransformer, DiEM, CNPFSPool
from utils import md5
from tqdm import tqdm
T = torch.Tensor


class NLL(nn.NLLLoss):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, ll, _):
        return -ll.mean()


class TestParallelMBC(object):
    def __init__(self) -> None:
        pass
    
    def test_celeba_memory_usage(self, bs=1) -> None:
        gpu = torch.cuda.current_device()
        
        st = partial(
            SetTransformer,
            x_dim=2, y_dim=3, d_dim=128,
            h_dim=128, d_hat=128,
            K=1, e_depth=4,
            d_depth=4, heads=1,
            ln_slots=True, ln_after=True,
            slot_type="random", slot_drop=0.0,
            attn_act="softmax", slot_residual=False
        )

        umbc = partial(
            UMBC, 
            x_dim=2, y_dim=3, d_dim=128,
            h_dim=128, d_hat=128,
            K=128, e_depth=4,
            d_depth=4, heads=1,
            ln_slots=True, ln_after=True,
            slot_type="random", slot_drop=0.0,
            attn_act="softmax", slot_residual=False
        )

        ds = partial(
            DeepSet, x_dim=2, y_dim=3, d_dim=128,
            h_dim=128, d_hat=128,
            K=1, e_depth=4,
            d_depth=4, heads=1,
            ln_slots=False, ln_after=False,
            slot_type="random", slot_drop=0.0,
            attn_act="softmax", slot_residual=False
        )
        fspool = partial(
            CNPFSPool, x_dim=2, y_dim=3, d_dim=128,
            h_dim=128, d_hat=128,
            K=1, e_depth=4,
            d_depth=4, heads=1,
            ln_slots=False, ln_after=False,
            slot_type="random", slot_drop=0.0,
            attn_act="softmax", slot_residual=False
        )

        diffem = partial(
            DiEM, x_dim=2, y_dim=3, d_dim=128,
            h_dim=128, d_hat=128,
            K=1, e_depth=4,
            d_depth=4, heads=1,
            ln_slots=True, ln_after=True,
            slot_type="random", slot_drop=0.0,
            attn_act="softmax", slot_residual=False
        )

        sse = partial(
            MBC, x_dim=2, y_dim=3, d_dim=128,
            h_dim=128, d_hat=128,
            K=1, e_depth=4,
            d_depth=4, heads=1,
            ln_slots=True, ln_after=True,
            slot_type="random", slot_drop=0.0,
            attn_act="slot-sigmoid", slot_residual=False
        )


        for mdl, name in tqdm(zip(
            [st, umbc, ds, sse, fspool, diffem],
            ["st", "umbc", "ds", "sse", "fspool","diffem"]
        ), position=0, leave=False):
            mem_used = []  # type: ignore
            for gc_size in tqdm([100, 200, 300, 400, 500], position=1, leave=False):
                model = mdl().to(gpu)
                loss_fn = NLL()
                d = {}

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
                target_x = torch.randn(bs, 1024, 2).to(gpu)
                target_y = torch.randn(bs, 1024, 3).to(gpu)

                if name in ["st", "fspool", "diffem"]:
                    context_set = torch.randn(bs, gc_size, 5).to(gpu)
                    out = model(context_set, target_x, target_y)
                else:
                    if gc_size > 100:
                        grad_c = torch.randn(bs, 100, 5).to(gpu)
                        no_grad_c = torch.randn(bs, gc_size-100, 5).to(gpu)
                        
                        out = model.partitioned_forward(grad_c, no_grad_c, target_x, target_y)
                    else:
                        context_set = torch.randn(bs, gc_size, 5).to(gpu)
                        out = model.partitioned_forward(context_set, None, target_x, target_y)
                loss = loss_fn(out, torch.zeros_like(out))
                total = 0
                for _, v in d.items():
                    total += 4 * v.numel()

                mem_used.append(total / 1000)
                loss.backward()

                for item in [model, context_set, target_x, target_y, loss, out]:
                    del item

            print(f"Model {name}: {mem_used}")

        # N = 10
        # lst: Any = []
        # lst.extend([(st, "st") for _ in range(N)])
        # lst.extend([(umbc, "umbc") for _ in range(N)])
        # lst.extend([(ds, "ds") for _ in range(N)])
        # lst.extend([(sse, "sse") for _ in range(N)])
        # random.shuffle(lst)

        # times = {"st": [], "umbc": [], "ds": [], "sse": []}
        # for mdl, name in tqdm(lst):
        #     model = mdl().to(gpu)

        #     start = datetime.now()
        #     target_x = torch.randn(bs, 1024, 2).to(gpu)
        #     target_y = torch.randn(bs, 1024, 3).to(gpu)

        #     if name == "st":
        #         context_set = torch.randn(bs, gc_size, 5).to(gpu)
        #         out = model(context_set, target_x, target_y)
        #     else:
        #         if gc_size > 100:
        #             grad_c = torch.randn(bs, 100, 5).to(gpu)
        #             no_grad_c = torch.randn(bs, gc_size-100, 5).to(gpu)
                    
        #             out = model.partitioned_forward(grad_c, no_grad_c, target_x, target_y)
        #         else:
        #             context_set = torch.randn(bs, gc_size, 5).to(gpu)
        #             out = model(context_set, target_x, target_y)
        #     loss = loss_fn(out, torch.zeros_like(out))
        #     times[name].append((datetime.now() - start).total_seconds())

        # for k in times.keys():
        #     t = np.array(times[k])
        #     print(f"{k} {t.mean()=} {t.std()=}")

if __name__ =="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--bs", type=int, default=1)
    args = parser.parse_args()
    tester = TestParallelMBC()
    tester.test_celeba_memory_usage(args.bs)
