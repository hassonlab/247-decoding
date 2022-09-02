import argparse
from turtle import color

import matplotlib.pyplot as plt
import pandas as pd

from aggregate_results import get_results


def plot(args):
    # Either read in the given results or generate them on the fly
    if args.input is not None:
        df = pd.read_csv(args.input)
    else:
        df = get_results()

    NUM_COLORS = 16
    cm = plt.get_cmap("gist_rainbow")
    colors = [cm(1.0 * i / NUM_COLORS) for i in range(NUM_COLORS)]
    cmap = {}
    for label, color in zip(args.labels, colors):
        cmap[label] = color

    # Plot each query as a separate line
    for query, label in zip(args.q, args.labels):
        sub_df = df.query(query).sort_values(by=args.x)
        x = sub_df[args.x]  # NOTE - hardcoded.
        y = sub_df[args.y]
        if "classify" in query:
            y = sub_df["avg_test_rocauc_test_w_avg"]

        assert not y.isna().all(), "all nans"

        if args.yerr is not None:
            yerr = sub_df[args.yerr]
            plt.fill_between(x, y - yerr, y + yerr, alpha=0.2)

        if args.label is not None:
            label = sub_df.iloc[0][args.label]

        plt.plot(x, y, marker=".", label=label, color=cmap[label])

    plt.grid()
    plt.legend(loc="upper left")
    plt.xlabel(args.x)
    plt.ylabel(args.y)
    plt.savefig(args.output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--q", nargs="+", required=True)
    parser.add_argument("--x", type=str, required=True)
    parser.add_argument("--y", type=str, required=True)
    parser.add_argument("--yerr", type=str, default=None)
    parser.add_argument("--label", type=str, default=None)
    parser.add_argument("--labels", nargs="*", type=str, default=[None] * 20)
    parser.add_argument("--input", type=str, default="results/aggregate.csv")
    parser.add_argument("--output", type=str, default="results/plots/out.png")

    args = parser.parse_args()
    plot(args)
