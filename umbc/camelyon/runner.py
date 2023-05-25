import os
from typing import Any

from utils import ENVLIST, Runner, remove_dups


def remove_duplicates(envs: ENVLIST) -> ENVLIST:
    cont = True
    while cont:
        envs_new = remove_dups(envs)
        if len(envs_new) == len(envs):
            cont = False
        envs = envs_new
    return envs


def get_env(
    run: int = 1,
    model: str = "sse",
    grad_set_size: int = 256,
    grad_correction: bool = True,
    pool: str = "max",
    mode: str = "pretrain",
    attn_act: str = "softmax",
    k: int = 64,
    patch_drop: float = 0.0,
    augmentation: bool = False,
    linear: bool = False,
) -> Any:

    env = os.environ.copy()
    env["ATTN_ACT"] = attn_act
    env["MODE"] = mode
    env["RUN"] = str(run)
    env["MODEL"] = model
    env["GRAD_SET_SIZE"] = str(grad_set_size)
    env["POOL"] = pool
    env["K"] = str(k)
    env["PATCH_DROPOUT"] = str(patch_drop)
    env["AUG"] = str(augmentation)
    env["LINEAR"] = str(linear)

    # this is set by utils runner but since this script
    # can use both, it needs to have a stock value which is
    # passed to the shell script in case we are using
    # single gpus in pretraining.
    env["GPUS"] = "0"
    env["GPU"] = "0"
    return env


def get_runs(
    model: str,
    mode: str,
    runs: int = 5,
    patch_drop: float = 0.0,
    augmentation: bool = False,
    linear: bool = False,
) -> ENVLIST:
    out_envs: ENVLIST = []

    if model in ["deepsets", "ds-mil", "ab-mil"]:
        for run in range(runs):
            out_envs.append(
                get_env(
                    model=model,
                    run=run,
                    mode=mode,
                    patch_drop=patch_drop,
                    augmentation=augmentation,
                    linear=linear,
                )
            )
    elif model in ["sse-umbc"]:
        for run in range(runs):
            # acts = ["slot-sigmoid", "slot-softmax", "slot-exp", "sigmoid"]
            acts = ["softmax"]
            for attn_act in acts:
                out_envs.append(
                    get_env(
                        model=model,
                        run=run,
                        mode=mode,
                        patch_drop=patch_drop,
                        attn_act=attn_act,
                        augmentation=augmentation,
                        linear=linear,
                    )
                )
    elif model == "sse":
        for run in range(runs):
            # acts = ["softmax", "slot-softmax", "slot-exp", "sigmoid"]
            acts = ["slot-sigmoid"]
            for attn_act in acts:
                out_envs.append(
                    get_env(
                        model=model,
                        run=run,
                        mode=mode,
                        attn_act=attn_act,
                        patch_drop=patch_drop,
                        augmentation=augmentation,
                        linear=linear,
                    )
                )
        # for run in range(runs):
        #     out_envs.append(
        #         get_env(
        #             model=model, run=run,
        #             mode=mode, attn_act="slot-sigmoid", patch_drop=patch_drop)
        #     )

    out_envs = remove_duplicates(out_envs)
    return out_envs


def do_runs(mode: str, gpus: Any) -> None:  # type: ignore
    envs = []
    patch_drop = 0.0
    aug = False
    linear = False
    # envs += get_runs("sse-umbc", mode, runs=5,
    #                  patch_drop=patch_drop, augmentation=aug, linear=linear)
    # envs += get_runs("ds-mil", mode, runs=5, patch_drop=patch_drop)
    # envs += get_runs("ab-mil", mode, runs=5, patch_drop=patch_drop)
    envs += get_runs("sse", mode, runs=5, patch_drop=patch_drop)
    # envs += get_runs("deepsets", mode, runs=5,
    #                  patch_drop=patch_drop, augmentation=aug, linear=linear)

    print(f"runs: {len(envs)}")
    runner = Runner("./run.sh", gpus, envs, test_after_train=False)
    runner.run()


if __name__ == "__main__":
    # this will include the testing loop
    # do_runs("pretrain", gpus=[0, 1, 2, 3, 4, 5, 6, 7] * 2)
    # do_runs("pretrain-test", gpus=[0, 1, 2, 3, 4, 5, 6, 7])

    do_runs("finetune", gpus=[[1], [2], [3], [4], [5]])
    # do_runs("finetune-test", gpus=[[2, 3, 4, 5, 6, 7]])
