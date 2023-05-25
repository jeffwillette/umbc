import os
from typing import Any, List

from utils import ENVLIST, Runner, remove_dups


def remove_duplicates(envs: ENVLIST) -> ENVLIST:
    cont = True
    while cont:
        envs_new = remove_dups(envs)
        if len(envs_new) == len(envs):
            cont = False
        envs = envs_new
    return envs


grad_train_same = (
    (8, 16, 32, 64, 128, 256, 512),
    (8, 16, 32, 64, 128, 256, 512)
)

grad_train_diff = (
    (8, 16, 32, 64, 128, 256),
    (512, 512, 512, 512, 512, 512)
)


def get_env(
    run: int = 1,
    model: str = "sse",
    grad_set_size: int = 1024,
    train_set_size: int = 1024,
    grad_correction: bool = True,
    pool: str = "mean",
    attn_act: str = "softmax"
) -> Any:

    env = os.environ.copy()
    env["ATTN_ACT"] = attn_act
    env["MODE"] = "mbc-motivation-example"
    env["RUN"] = str(run)
    env["MODEL"] = model
    env["EPOCHS"] = str(50)
    env["SLOT_TYPE"] = "random"
    env["HEADS"] = str(4)
    env["GRAD_CORRECTION"] = str(grad_correction)
    env["GRAD_SET_SIZE"] = str(grad_set_size)
    env["TRAIN_SET_SIZE"] = str(train_set_size)
    env["POOL"] = pool

    return [env]


def get_set_transformer_runs(
    runs: int = 5,
    n_parallel_list: List[int] = [1]
) -> ENVLIST:

    out_envs: ENVLIST = []
    model = "set-transformer"

    for run in range(runs):
        # the baseline runs with grad set size equal to the train set size
        out_envs.extend(get_env(
            model=model,
            run=run,
            grad_set_size=8,
            train_set_size=8,
            grad_correction=False,
        ))

    out_envs = remove_duplicates(out_envs)
    return out_envs


def get_deepsets_runs(
    runs: int = 5,
    n_parallel_list: List[int] = [1]
) -> ENVLIST:

    out_envs: ENVLIST = []
    model = "deepsets"

    for run in range(runs):
        # the baseline runs with grad set size equal to the train set size
        out_envs.extend(get_env(
            model=model,
            run=run,
            grad_set_size=8,
            train_set_size=512,
            grad_correction=True,
        ))

    out_envs = remove_duplicates(out_envs)
    return out_envs


def get_umbc_runs(runs: int = 5, n_parallel_list: List[int] = [1]) -> ENVLIST:
    out_envs: ENVLIST = []
    model = "sse-umbc"

    for run in range(runs):
        # the baseline runs with grad set size equal to the train set size
        out_envs.extend(get_env(
            model=model,
            run=run,
            grad_set_size=8,
            train_set_size=512,
            grad_correction=True,
            attn_act="softmax"
        ))

    out_envs = remove_duplicates(out_envs)
    return out_envs


def get_sse_runs(runs: int = 5, n_parallel_list: List[int] = [1]) -> ENVLIST:
    out_envs: ENVLIST = []
    model = "sse"

    for run in range(runs):
        # the baseline runs with grad set size equal to the train set size
        out_envs.extend(get_env(
            model=model,
            run=run,
            grad_set_size=8,
            train_set_size=512,
            grad_correction=True,
            attn_act="slot-sigmoid"
        ))

    out_envs = remove_duplicates(out_envs)
    return out_envs


def do_motivation_runs() -> None:  # type: ignore
    envs: ENVLIST = []
    RUNS = 1
    GPUS = [1] * 3

    envs += get_umbc_runs(RUNS)
    envs += get_set_transformer_runs(RUNS)
    envs += get_sse_runs(RUNS)
    envs += get_deepsets_runs(RUNS)

    train_runs, test_runs = [v for v in envs if v["MODE"] != "test"], [
        v for v in envs if v["MODE"] == "test"]

    print(f"training: {len(train_runs)}")
    print(f"test: {len(test_runs)}")

    runner = Runner("./run.sh", GPUS, train_runs, test_after_train=False)
    runner.run()


if __name__ == "__main__":
    do_motivation_runs()
