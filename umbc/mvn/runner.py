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

    envs: ENVLIST = []
    for mode in ["train", "test"]:
        env = os.environ.copy()
        env["ATTN_ACT"] = attn_act
        env["MODE"] = mode
        env["RUN"] = str(run)
        env["MODEL"] = model
        env["EPOCHS"] = str(50)
        env["SLOT_TYPE"] = "random"
        env["HEADS"] = str(4)
        env["GRAD_CORRECTION"] = str(grad_correction)
        env["GRAD_SET_SIZE"] = str(grad_set_size)
        env["TRAIN_SET_SIZE"] = str(train_set_size)
        env["POOL"] = pool

        envs.append(env)
    return envs


def get_set_transformer_runs(
    runs: int = 5,
    n_parallel_list: List[int] = [1]
) -> ENVLIST:

    out_envs: ENVLIST = []
    model = "set-transformer"

    for run in range(runs):
        # the baseline runs with grad set size equal to the train set size
        for grad_set_size, train_set_size in zip(*grad_train_same):
            out_envs.extend(get_env(
                model=model,
                run=run,
                grad_set_size=grad_set_size,
                train_set_size=train_set_size,
                grad_correction=False,
            ))

    out_envs = remove_duplicates(out_envs)
    return out_envs


def get_diffem_runs(
    runs: int = 5,
    n_parallel_list: List[int] = [1]
) -> ENVLIST:

    out_envs: ENVLIST = []
    model = "diff-em"

    for run in range(runs):
        # the baseline runs with grad set size equal to the train set size
        for grad_set_size, train_set_size in zip(*grad_train_same):
            out_envs.extend(get_env(
                model=model,
                run=run,
                grad_set_size=grad_set_size,
                train_set_size=train_set_size,
                grad_correction=False,
            ))

            print("BREAKING TO ONLY DO THE SMALL RUN FOR DIFFEM")
            break

    out_envs = remove_duplicates(out_envs)
    return out_envs


def get_fspool_runs(
    runs: int = 5,
    n_parallel_list: List[int] = [1]
) -> ENVLIST:

    out_envs: ENVLIST = []
    model = "fspool"

    for run in range(runs):
        # the baseline runs with grad set size equal to the train set size
        for grad_set_size, train_set_size in zip(*grad_train_same):
            out_envs.extend(get_env(
                model=model,
                run=run,
                grad_set_size=grad_set_size,
                train_set_size=train_set_size,
                grad_correction=False,
            ))

            print("BREAKING TO ONLY DO THE SMALL RUN FOR FSPool")
            break

    out_envs = remove_duplicates(out_envs)
    return out_envs


def get_deepsets_runs(
    runs: int = 5,
    n_parallel_list: List[int] = [1]
) -> ENVLIST:

    out_envs: ENVLIST = []
    model = "deepsets"
    pool_funcs = ["min", "max", "mean", "sum"]

    for run in range(runs):
        # the baseline runs with grad set size equal to the train set size
        for grad_set_size, train_set_size in zip(*grad_train_same):
            out_envs.extend(get_env(
                model=model,
                run=run,
                grad_set_size=grad_set_size,
                train_set_size=train_set_size,
                grad_correction=False,
            ))

        # runs which will evaluate the gradient correction/gradient set size
        for grad_correction in [False, True]:
            for grad_set_size, train_set_size in zip(*grad_train_diff):
                out_envs.extend(get_env(
                    model=model,
                    run=run,
                    grad_set_size=grad_set_size,
                    train_set_size=train_set_size,
                    grad_correction=grad_correction,
                ))

        for pool in pool_funcs:
            out_envs.extend(get_env(
                model=model,
                run=run,
                grad_set_size=8,
                train_set_size=1024,
                grad_correction=True,
            ))

    out_envs = remove_duplicates(out_envs)
    return out_envs


def get_umbc_diffem_runs(runs: int = 5, n_parallel_list: List[int] = [1]) -> ENVLIST:
    out_envs: ENVLIST = []
    model = "diff-em-umbc"

    for run in range(runs):
        # runs which will evaluate the gradient correction/gradient set size
        for grad_set_size, train_set_size in zip(*grad_train_diff):
            out_envs.extend(get_env(
                model=model,
                run=run,
                grad_set_size=grad_set_size,
                train_set_size=train_set_size,
                grad_correction=True,
            ))
            break

    out_envs = remove_duplicates(out_envs)
    return out_envs


def get_umbc_fspool_runs(runs: int = 5, n_parallel_list: List[int] = [1]) -> ENVLIST:
    out_envs: ENVLIST = []
    model = "fspool-umbc"

    for run in range(runs):
        # runs which will evaluate the gradient correction/gradient set size
        for grad_set_size, train_set_size in zip(*grad_train_diff):
            out_envs.extend(get_env(
                model=model,
                run=run,
                grad_set_size=grad_set_size,
                train_set_size=train_set_size,
                grad_correction=True,
            ))
            break

    out_envs = remove_duplicates(out_envs)
    return out_envs


def get_umbc_runs(runs: int = 5, n_parallel_list: List[int] = [1]) -> ENVLIST:
    out_envs: ENVLIST = []
    model = "sse-umbc"
    acts = ["sigmoid", "softmax", "slot-softmax", "slot-sigmoid", "slot-exp"]

    for run in range(runs):
        # the baseline runs with grad set size equal to the train set size
        for grad_set_size, train_set_size in zip(*grad_train_same):
            out_envs.extend(get_env(
                model=model,
                run=run,
                grad_set_size=grad_set_size,
                train_set_size=train_set_size,
                grad_correction=False,
            ))

        # runs which will evaluate the gradient correction/gradient set size
        for grad_correction in [False, True]:
            for grad_set_size, train_set_size in zip(*grad_train_diff):
                out_envs.extend(get_env(
                    model=model,
                    run=run,
                    grad_set_size=grad_set_size,
                    train_set_size=train_set_size,
                    grad_correction=grad_correction,
                ))

        for attn_act in acts:
            out_envs.extend(get_env(
                model=model,
                run=run,
                grad_set_size=8,
                train_set_size=512,
                grad_correction=True,
                attn_act=attn_act
            ))

    out_envs = remove_duplicates(out_envs)
    return out_envs


def get_sse_runs(runs: int = 5, n_parallel_list: List[int] = [1]) -> ENVLIST:
    out_envs: ENVLIST = []
    model = "sse"
    acts = ["sigmoid", "softmax", "slot-softmax", "slot-sigmoid", "slot-exp"]

    for run in range(runs):
        # the baseline runs with grad set size equal to the train set size
        for grad_set_size, train_set_size in zip(*grad_train_same):
            out_envs.extend(get_env(
                model=model,
                run=run,
                grad_set_size=grad_set_size,
                train_set_size=train_set_size,
                grad_correction=False,
                attn_act="slot-sigmoid"
            ))

        # runs which will evaluate the gradient correction/gradient set size
        for grad_correction in [False, True]:
            for grad_set_size, train_set_size in zip(*grad_train_diff):
                out_envs.extend(get_env(
                    model=model,
                    run=run,
                    grad_set_size=grad_set_size,
                    train_set_size=train_set_size,
                    grad_correction=grad_correction,
                    attn_act="slot-sigmoid"
                ))

        for attn_act in acts:
            out_envs.extend(get_env(
                model=model,
                run=run,
                grad_set_size=8,
                train_set_size=512,
                grad_correction=True,
                attn_act=attn_act
            ))

    out_envs = remove_duplicates(out_envs)
    return out_envs


def get_sse_hierarchical_runs(
    runs: int = 5,
    n_parallel_list: List[int] = [1]
) -> ENVLIST:
    out_envs: ENVLIST = []
    model = "sse-hierarchical"

    for run in range(runs):
        # runs which will evaluate the gradient correction/gradient set size
        for grad_set_size, train_set_size in zip(*grad_train_diff):
            out_envs.extend(get_env(
                model=model,
                run=run,
                grad_set_size=grad_set_size,
                train_set_size=train_set_size,
                grad_correction=True,
                attn_act="slot-sigmoid",
            ))
            break

    out_envs = remove_duplicates(out_envs)
    return out_envs


def do_training_runs() -> None:  # type: ignore
    envs: ENVLIST = []
    RUNS = 5
    # GPUS = [0, 1, 2, 3, 4, 5] * 3
    GPUS = [6, 7] * 2

    # envs += get_umbc_runs(RUNS)
    # envs += get_set_transformer_runs(RUNS)
    # envs += get_sse_runs(RUNS)
    # envs += get_deepsets_runs(RUNS)
    # envs += get_fspool_runs(RUNS)
    # envs += get_diffem_runs(RUNS)
    # envs += get_umbc_fspool_runs(RUNS)
    # envs += get_umbc_diffem_runs(RUNS)
    envs += get_sse_hierarchical_runs(RUNS)

    train_runs, test_runs = [v for v in envs if v["MODE"] != "test"], [
        v for v in envs if v["MODE"] == "test"]

    print(f"training: {len(train_runs)}")
    print(f"test: {len(test_runs)}")

    train_runner = Runner("./run.sh", GPUS, train_runs, test_after_train=False)
    train_runner.run()

    test_runner = Runner("./run.sh", GPUS, test_runs, test_after_train=False)
    test_runner.run()


if __name__ == "__main__":
    do_training_runs()
