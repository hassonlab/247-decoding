import argparse
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import glob
import os

from aggregate_results import get_results
from statsmodels.stats import multitest
from scipy.stats import permutation_test

def fdr(pvals):
    """p-value correction for multiple tests using Benjamini/Hochberg false discovery rate method""" 
    _, pcor, _, _ = multitest.multipletests(pvals,method='fdr_bh',is_sorted=False)
    return pcor 

def one_samp_perm(x, nperms):

  n = len(x)
  dist = np.zeros(nperms)
  for i in range(nperms):
      dist[i] = np.random.choice(x, n, replace=True).mean()
  val = np.sum(dist > 0.5) # 0.5 is specific for AUCROC
  p_value = 1 - val / nperms
  return p_value

def paired_permutation(x, y, nperms):

    n = len(x)
    truescore = (x - y).mean()
    dist = np.zeros(nperms)
    for i in range(nperms):
        s = np.random.choice([1, -1], n)
        dist[i] = np.mean(s * (x-y))
    p_value = (truescore > dist).mean()
    return p_value


def plot(args):
    sub_id = args.sig.split('-')[1]
    nperms = 10000
    # Either read in the given results or generate them on the fly
    if args.input is not None:
        df = pd.read_csv(args.input)
    else:
        df = get_results()


    lag_folders = glob.glob(os.path.join("results", args.sig, "*"))
    lag_names = sorted([int(os.path.basename(lag_folder)) for lag_folder in lag_folders])
    sorted_lag_folders = [os.path.join("results", args.sig, str(lag_name)) for lag_name in lag_names]
   
    pair_pvals = []
    ifg_pvals = []
    mean_aucrocs = []
    std_aucrocs = []
    for lag_folder in sorted_lag_folders:
    
        csv_1 = os.path.join(lag_folder, "ensemble", "avg_test_topk_rocaauc_df.csv")

        results = pd.read_csv(csv_1)
        aucrocs = results['rocauc'].values
        ifg_pval = one_samp_perm(aucrocs, nperms)
        ifg_pvals.append(ifg_pval)
        mean_aucrocs.append(aucrocs.mean())
        csv_2 = csv_1.replace("IFG","PRE")
        results2 = pd.read_csv(csv_2)
        aucrocs2 = results2['rocauc'].values 
        pair_pval = paired_permutation(aucrocs2, aucrocs, nperms)
        pair_pvals.append(pair_pval)
    
    pair_pcorrs = fdr(pair_pvals)
    ifg_pcorrs = fdr(ifg_pvals)

    mean_aucrocs = np.array(mean_aucrocs)
    minP = 0
    g = ifg_pcorrs <= 0.01

    if g.any():
        sigSig = mean_aucrocs[g]
        if (ifg_pcorrs > 0.01).any():
            maxP = mean_aucrocs[ifg_pcorrs > 0.01].max()
            gg = sigSig > maxP
            if gg.any():
                minP = sigSig[gg].min()

    issig = (mean_aucrocs > minP) & (pair_pcorrs < 0.01)
    siglags = issig.nonzero()[0]

    # Plot each query as a separate line
    for query, label in zip(args.q, args.labels):

        sub_df = df.query(query).sort_values(by=args.x)
        x = sub_df[args.x]
        y = sub_df[args.y]
        assert not y.isna().all(), "all nans"

        if args.label is not None:
            label = sub_df.iloc[0][args.label]
        if label == "IFG":
            color = "darkviolet"
        elif label == "Pre":
            color = "limegreen"
        elif label == "IFG-sh":
            color = "#190136"
        elif label == "Pre-sh":
            color = "#013611"   
        
        if args.yerr is not None:
            lags = np.arange(-4000,4100,100)
            yem2 = []
            for lag in lags:
                lag_folder = os.path.join("results", sub_df.model.unique()[0], str(lag))
                csv_1 = os.path.join(lag_folder, "ensemble", "avg_test_topk_rocaauc_df.csv")
                results = pd.read_csv(csv_1).rocauc
                yem2.append(results.sem())
            plt.fill_between(x, y - yem2, y + yem2, alpha=0.2, color = color)
        plt.plot(x, y, label=label, color=color)

    lags = np.array(lag_names)
    if sub_id == '717':
        plt.title('Participant 1', fontweight='bold')
    elif sub_id == '742':
        plt.title('Participant 2', fontweight='bold')
    elif sub_id == '798':
        plt.title('Participant 3', fontweight='bold')
    elif sub_id =='all3':
        plt.title('All Participants', fontweight='bold')

    plt.legend(loc="upper left")
    plt.xlabel('Time (ms)', fontweight='bold')
    plt.ylabel("ROC-AUC", fontweight='bold')
    plt.ylim(0.45, 0.72)
    plt.yticks([0.5, 0.55, 0.6, 0.65, 0.7])
    plt.axhline(minP)
    plt.scatter(lags[siglags], [0.70]*len(siglags), marker ='*', color = 'darkviolet')
    plt.savefig(args.output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--q", nargs="+", required=True)
    parser.add_argument("--x", type=str, required=True)
    parser.add_argument("--y", type=str, required=True)
    parser.add_argument("--yerr", type=str, default=None)
    parser.add_argument("--label", type=str, default=None)
    parser.add_argument("--sig", type=str, default=None)
    parser.add_argument("--labels", nargs="*", type=str, default=[None] * 10)
    parser.add_argument("--input", type=str, default="results/aggregate.csv")
    parser.add_argument("--output", type=str, default="results/plots/out.png")

    args = parser.parse_args()
    plot(args)
