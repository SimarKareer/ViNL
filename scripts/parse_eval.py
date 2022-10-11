import argparse
import glob
import json
import os.path as osp

import numpy as np
import pandas as pd

METRICS = [
    "dist_traveled",
    "feet_collisions",
    "feet_collisions_per_meter",
    "success",
    "num_steps",
]
KEYS = ["map_name", "episode_id", "seed", "attempt"] + METRICS


def main(eval_dir):
    nice_metrics = ["success", "dist_traveled", "feet_collisions_per_meter"]
    files = glob.glob(osp.join(eval_dir, "*.txt"))
    if files:
        file_parser, num_keys, num_metrics = parse_txt, len(KEYS) - 1, len(METRICS) - 1
    else:
        file_parser, num_keys, num_metrics = parse_json, len(KEYS), len(METRICS)
        files = glob.glob(osp.join(eval_dir, "*.json"))
        nice_metrics.append("feet_collisions_per_step")

    # Convert dict to pandas dataframe
    df = generate_df(files, num_keys=num_keys, parser_fn=file_parser)

    # Filter out episodes that were too short
    df = df[df["dist_traveled"] > 1]

    # Group by seed (attempt) and get mean and std of mean across attempts
    num_attempts = max(df["attempt"]) + 1
    aggregated_stats = {
        k: [] for k in METRICS[:num_metrics] + ["feet_collisions_per_step"]
    }
    for idx in range(int(num_attempts)):
        filtered_df = df[df["attempt"] == idx]
        for k in METRICS[:num_metrics]:
            aggregated_stats[k].append(np.mean(filtered_df[k]))
        if "feet_collisions_per_step" in nice_metrics:
            feet_collisions_per_step = np.array(
                filtered_df["feet_collisions"]
            ) / np.array(filtered_df["num_steps"])
            aggregated_stats["feet_collisions_per_step"].append(
                np.mean(feet_collisions_per_step)
            )

    row = ""
    latex_row = ""
    for m in nice_metrics:
        scale = 100 if m in ["success", "feet_collisions_per_step"] else 1
        latex_str = (
            f"{np.mean(aggregated_stats[m])*scale:.2f} "
            f"$\pm$ {np.std(aggregated_stats[m])*scale:.2f} & "
        )
        latex_row += latex_str
        if m != "dist_traveled":  # Don't care about dist for Google Sheet
            row += latex_str
    print("\t".join(nice_metrics))
    print(latex_row[:-2] + "\\\\")
    print(row[:-2].replace(" $\pm$ ", "+-").replace(" & ", "\t"))


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


def parse_json(file):
    with open(file) as f:
        data = json.load(f)
    return [data[KEYS[0]]] + [float(data[k]) for k in KEYS[1:]]


def generate_df(files, num_keys=len(KEYS), parser_fn=parse_json):
    data_dict = {k: [] for k in KEYS[:num_keys]}
    for file in files:
        data = parser_fn(file)
        assert num_keys == len(data)
        for k, v in zip(KEYS[:num_keys], data):
            data_dict[k].append(v)
    print(f"Num episodes: {len(data_dict['map_name'])}")

    # Convert dict to pandas dataframe
    df = pd.DataFrame.from_dict(data_dict)
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("eval_dir")
    args = parser.parse_args()
    main(args.eval_dir)
