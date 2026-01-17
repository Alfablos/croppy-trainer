import time
from pathlib import Path
from time import sleep
from typing import List

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pyplot import legend
from tqdm import tqdm


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
    with tqdm(total=10, leave=True, desc="Sleeping") as bar:
        while True:
            plot_csvs(list(root.glob("**/compute_durations_*.csv", case_sensitive=True)))
            for _ in range(10):
                sleep(1)
                bar.update(1)