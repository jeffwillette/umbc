import os
import queue
import subprocess
import time
from typing import Any, Dict, List, Tuple, Union

from utils import get_module_root, set_logger

ENVLIST = List[Dict[str, str]]
PROCLIST = List[Tuple[subprocess.Popen, int, Dict[str, str]]]
GPULIST = Union[List[int], List[List[int]]]
GPU = Union[int, List[int]]


def get_env(**kwargs: Any) -> Dict[str, str]:
    env = os.environ.copy()
    for k in kwargs:
        env[k.upper()] = str(kwargs[k])

    return env


def get_gpu_string(gpu: GPU) -> Tuple[str, str]:
    if isinstance(gpu, list):
        return "GPUS", " ".join([str(v) for v in gpu])
    return "GPU", str(gpu)


class Runner:
    def __init__(
        self,
        script: str,
        gpus: GPULIST,
        envs: ENVLIST,
        test_after_train: bool = True
    ) -> None:
        self.script = script
        self.gpus = gpus
        self.log = set_logger("INFO")
        self.envs = envs
        self.test_after_train = test_after_train

        self.procs: PROCLIST = []
        self.gpu_q: queue.Queue = queue.LifoQueue()
        for n in self.gpus:
            self.gpu_q.put(n)

    def _check_procs(self) -> None:
        for j, (proc, gpu, env) in enumerate(self.procs):
            if proc.poll() is not None:
                # remove this one from the list of processes
                _, _, _ = self.procs.pop(j)

                if self.test_after_train and env["MODE"] == "train":
                    # after the training has finished, we want to run the
                    # tests. Add another test process and continue.
                    # Without freeing up the GPU.
                    self.log.info(f"process on gpu: {gpu} running tests")
                    env["MODE"] = "test"
                    proc = subprocess.Popen(
                        ["/bin/bash", self.script], env=env)
                    self.procs.append((proc, gpu, env))
                    return

                # If this flag is not set then we need to just put the GPU
                # back in the queue and continue
                self.log.info(
                    f"mode: {env['MODE']} on gpu: {gpu} finished. freeing GPU")
                self.gpu_q.put(gpu)
            else:
                time.sleep(1)

    def run(self) -> None:
        self.log.info("starting process loop")
        for i, env in enumerate(self.envs):
            gpu = self.gpu_q.get()

            env["PYTHONPATH"] = get_module_root()
            VAR, VAL = get_gpu_string(gpu)
            env[VAR] = VAL

            proc = subprocess.Popen(["/bin/bash", self.script], env=env)

            self.procs.append((proc, gpu, env))
            self.log.info(f"added process: {i} on gpu: {gpu}")
            while self.gpu_q.empty():
                # if there was no testing after training, procs will be empty
                if len(self.procs) == 0:
                    break
                self._check_procs()

        # at the end, the gpu queue will not be empty but we will still have
        # running processes, therefore we have to wait for the processes
        # in the procs list to finish before exiting
        while len(self.procs) > 0:
            self._check_procs()

        self.log.info("finished")
