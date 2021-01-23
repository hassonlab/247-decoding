import argparse

import matplotlib.pyplot as plt
import pandas as pd

from aggregate_results import get_results


def plot(args):
    # Either read in the given results or generate them on the fly
    if args.input is not None:
        df = pd.read_csv(args.input)
    else:
        df = get_results()

    # Plot each query as a separate line
    for query in args.q:
        sub_df = df.query(query).sort_values(by=args.x)
        x = sub_df[args.x] / 512  # NOTE - hardcoded.
        y = sub_df[args.y]

        if args.yerr is not None:
            yerr = sub_df[args.yerr]
            plt.fill_between(x, y - yerr, y + yerr, alpha=0.2)
        plt.plot(x, y, marker='.')

    plt.xlabel(args.x)
    plt.ylabel(args.y)
    plt.grid()
    plt.savefig(args.output)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--q', nargs='+', required=True)
    parser.add_argument('--x', type=str, required=True)
    parser.add_argument('--y', type=str, required=True)
    parser.add_argument('--yerr', type=str, default=None)
    parser.add_argument('--input', type=str, default=None)
    parser.add_argument('--output', type=str, default='results/plots/out.png')

    args = parser.parse_args()
    plot(args)
