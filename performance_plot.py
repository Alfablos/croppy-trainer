import time
from pathlib import Path
from typing import List

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pyplot import legend


def plot_csvs(paths: List[Path]):
    fig, ax = plt.subplots(figsize=(10, 10))
    for p in paths:
        df = pd.read_csv(p)
        ax.plot(
            df["duration"],
            df["batch_size"],
            label=p.name
        )
        ax.legend()

    fig.savefig("test.png")



if __name__ == "__main__":
    root = Path("./")
    while True:
        plot_csvs(list(root.glob("**/compute_durations_*.csv", case_sensitive=True)))
        time.sleep(10)