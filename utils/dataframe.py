from typing import Any, List

import pandas as pd  # type: ignore


def df_files(filepaths: List[str]) -> pd.DataFrame:
    # read the last line of all the files and enforce that they are the same

    cols: Any = {}
    for i, path in enumerate(filepaths):
        with open(path, "r") as f:
            if i == 0:
                cols = {k: [] for k in f.readline().split(",")}
            else:
                s = f.readline().split(",")
                if not all([u == v for (u, v) in zip(cols.keys(), s)]) and len(s) != len(cols.keys()):
                    raise ValueError(f"each file needs to have the same stats: {cols.keys()=} {s=}")

            for j, line in enumerate(f):
                pass  # go to the last line of the file which are teh stats from the current run

            for stat, key in zip(line.split(","), cols.keys()):
                cols[key].append(float(stat))

    return pd.DataFrame(cols)
