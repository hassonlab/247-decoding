import argparse
import itertools

import matplotlib.pyplot as plt
import pandas as pd


def plot(args):
    # Either read in the given results or generate them on the fly
    df = pd.read_csv(args.input)

    colors = plt.cm.get_cmap('RdYlBu_r')

    # Plot each query as a separate line
    for query, label, value in itertools.zip_longest(args.q,
                                                     args.labels,
                                                     args.values):
        if value is not None:
            query = query % value
        sub_df = df.query(query).sort_values(by=args.x)
        x = sub_df[args.x]
        y = sub_df[args.y]

        if y.isna().all():
            print(f'Query did not result in any results:\n{query}')
            continue

        color = colors(value / len(args.values))

        if args.yerr is not None:
            yerr = sub_df[args.yerr]
            plt.fill_between(x, y - yerr, y + yerr, alpha=0.2, color=color)

        if args.label is not None:
            label = sub_df.iloc[0][args.label]

        plt.plot(x, y, marker='.', label=label, color=color)
        print(value, y.max())

    plt.grid()
    plt.legend(loc='upper left')
    plt.xlabel(args.x)
    plt.ylabel(args.y)
    plt.savefig(args.output)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--q', nargs='+', required=True)
    parser.add_argument('--x', type=str, required=True)
    parser.add_argument('--y', type=str, required=True)
    parser.add_argument('--yerr', type=str, default=None)
    parser.add_argument('--label', type=str, default=None)
    parser.add_argument('--values', nargs='*', type=str, default=[])
    parser.add_argument('--labels', nargs='*', type=str, default=[])
    parser.add_argument('--input', type=str, default='results/aggregate.csv')
    parser.add_argument('--output', type=str, default='results/plots/out.png')
    args = parser.parse_args()

    if not len(args.values):
        args.values = [None] * len(args.q)
    else:
        if args.values[0].isnumeric():
            args.values = [int(v) for v in args.values]
        args.q = [args.q[0]] * len(args.values)  # NOTE

    if not len(args.labels):
        args.labels = [None] * len(args.q)

    plot(args)
