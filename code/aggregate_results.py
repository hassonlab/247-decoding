import glob
import json

import pandas as pd
import os
from argparse import ArgumentParser


def get_results(args):
    results = []
    for filename in glob.glob(os.path.join(args.results,'*/*/*/results.json')):
        with open(filename, 'r') as fp:
            all_results = json.load(fp)

            result = {
                k: v
                for k, v in all_results.items() if k.startswith('avg')
            }
            result.update({
                k: v
                for k, v in all_results['args'].items() if type(v) != list
            })
            results.append(result)

    df = pd.DataFrame(results)
    return df


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--results', type=str, default='results')
    parser.add_argument('--output', type=str, default='results/aggregate.csv')
    args = parser.parse_args()

    df = get_results(args)
    df.to_csv(args.output, index=False)
