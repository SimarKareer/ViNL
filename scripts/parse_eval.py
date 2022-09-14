import argparse
import glob
import os.path as osp

import numpy as np
import pandas as pd

METRICS = [
    "dist_traveled",
    "feet_collisions",
    "feet_collisions_per_meter",
    "success",
]
KEYS = [
    "map_name",
    "episode_id",
    "seed",
    "attempt",
] + METRICS


def main(eval_dir):
    txt_files = glob.glob(osp.join(eval_dir, "*.txt"))
    data_dict = {k: [] for k in KEYS}
    for txt in txt_files:
        data = parse_txt(txt)
        assert len(KEYS) == len(data)
        for k, v in zip(KEYS, data):
            data_dict[k].append(v)
    print(f"Num episodes: {len(data_dict['map_name'])}")

    df = pd.DataFrame.from_dict(data_dict)
    num_attempts = max(df["attempt"])
    aggregated_stats = {k: [] for k in METRICS}
    for idx in range(int(num_attempts) + 1):
        for k in METRICS:
            filtered_df = df[df["attempt"] == idx]
            aggregated_stats[k].append(np.mean(filtered_df[k]))

    row = ""
    nice_metrics = [
        "success",
        "dist_traveled",
        "feet_collisions_per_meter",
        "feet_collisions",
    ]
    for m in nice_metrics:
        if m == "success":
            row += (
                f"{np.mean(aggregated_stats[m])*100:.2f} "
                f"$\pm$ {np.std(aggregated_stats[m])*100:.2f}\t"
            )
        else:
            row += (
                f"{np.mean(aggregated_stats[m]):.2f} "
                f"$\pm$ {np.std(aggregated_stats[m]):.2f}\t"
            )
    print("\t".join(nice_metrics))
    print(row[:-1] + " \\\\")


def parse_txt(txt):
    base = osp.basename(txt)[: -len(".txt")]
    map_name, episode_id, seed, attempt = base.split("_")
    episode_id, seed, attempt = float(episode_id), float(seed), float(attempt)
    with open(txt) as f:
        data = f.read().splitlines()
    data = sorted(data)

    (
        dist_traveled,
        feet_collisions,
        feet_collisions_per_meter,
        success,
    ) = [float(line.split()[-1]) for line in data if ":" in line]

    return (
        map_name,
        episode_id,
        seed,
        attempt,
        dist_traveled,
        feet_collisions,
        feet_collisions_per_meter,
        success,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("eval_dir")
    args = parser.parse_args()
    main(args.eval_dir)
