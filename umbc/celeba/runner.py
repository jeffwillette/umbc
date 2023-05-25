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


def get_runs(runs: int = 5, n_parallel_list: List[int] = [1]) -> ENVLIST:
    out_envs: ENVLIST = []

    def get_env(
        run: int = 1,
        model: str = "set-xformer",
        umbc_grad_size: int = 1024,
        train_set_size: int = 1024,
        grad_correction: bool = True
    ) -> Any:

        envs: ENVLIST = []
        for mode in ["train", "test"]:
            env = os.environ.copy()
            env["DATASET"] = "toy-mixture-of-gaussians"
            env["ATTN_ACT"] = "slot-sigmoid"
            env["MODE"] = mode
            env["RUN"] = str(run)
            env["MODEL"] = model
            env["EPOCHS"] = str(50)
            env["SLOT_TYPE"] = "random"
            env["UNIVERSAL"] = str(True)
            env["HEADS"] = str(4)
            env["FIXED"] = str(False)  # whether this model will have a fixed encoder
            env["SLOT_DROP"] = str(0.0)
            env["SLOT_RESIDUAL"] = str(True)
            env["GRAD_CORRECTION"] = str(grad_correction)
            env["UMBC_GRAD_SIZE"] = str(umbc_grad_size)
            env["TRAIN_SET_SIZE"] = str(train_set_size)
            env["LN_AFTER"] = str(True)
            env["UNIVERSAL_K"] = str(128)
            env["N_PARALLEL"] = str(1)

            envs.append(env)
        return envs

    for run in range(runs):
        # the baseline runs with grad set size equal to the train set size
        for umbc_grad_size, train_set_size in zip((32, 1024), (32, 1024)):
            out_envs.extend(get_env(
                run=run,
                umbc_grad_size=umbc_grad_size,
                train_set_size=train_set_size,
                grad_correction=False
            ))

        # runs which will evaluate the gradient correction and gradient set size
        for grad_correction in [False, True]:
            for umbc_grad_size, train_set_size in zip(
                (8, 16, 32, 64, 128, 256, 512),
                (1024, 1024, 1024, 1024, 1024, 1024, 1024)
            ):
                out_envs.extend(get_env(
                    run=run,
                    umbc_grad_size=umbc_grad_size,
                    train_set_size=train_set_size,
                    grad_correction=grad_correction
                ))

    out_envs = remove_duplicates(out_envs)
    return out_envs


def do_motivation_run() -> None:  # type: ignore
    envs: ENVLIST = []
    RUNS = 5
    parallel = [1]
    GPUS = [0, 1, 2, 3, 4, 5, 6, 7] * 2

    envs = get_runs(RUNS, n_parallel_list=parallel)
    train_runs, test_runs = [v for v in envs if v["MODE"] != "test"], [v for v in envs if v["MODE"] == "test"]
    print(f"training: {len(train_runs)}")
    print(f"test: {len(test_runs)}")

    finetune_runner = Runner("./run.sh", GPUS, train_runs, test_after_train=False)
    finetune_runner.run()

    test_runner = Runner("./run.sh", GPUS, test_runs, test_after_train=False)
    test_runner.run()


if __name__ == "__main__":
    do_motivation_run()
