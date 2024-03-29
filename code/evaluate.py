from collections import Counter, defaultdict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist
from scipy.special import softmax
from sklearn.metrics import auc, confusion_matrix, roc_curve, roc_auc_score


def corr(A, B):
    A = A - A.mean(axis=0).reshape(1, -1)
    B = B - B.mean(axis=0).reshape(1, -1)
    return np.sum(A * B, axis=0) / (
        np.sqrt(np.sum(A * A, axis=0) * np.sum(B * B, axis=0)) + 1e-7)


def pearson_r(A, B, axis=-1):
    '''Calculate pearson correlation between two matricies'''

    A_mean = A.mean(axis=axis).reshape(-1, 1)
    B_mean = B.mean(axis=axis).reshape(-1, 1)
    A_stddev = np.sum((A - A_mean)**2, axis=axis)
    B_stddev = np.sum((B - B_mean)**2, axis=axis)

    num = np.sum((A - A_mean) * (B - B_mean), axis=axis)
    den = np.sqrt(A_stddev * B_stddev)

    return num / den  # array of correlations for each observation
    # return (num / den).mean(dtype=np.float)


def evaluate_embeddings(y_true, y_pred, prefix, save_dir=None, suffix=''):
    ''' Compute pearson correlation between actual and predicted embeddings
       both along row (word) and column (feature) axes'''
    r = corr(y_pred, y_true)
    rT = corr(y_pred.T, y_true.T)

    if save_dir is not None:
        fig, axes = plt.subplots(1, 3, figsize=(21, 6))
        axes[0].bar(np.arange(len(r)), r)
        axes[0].set_xlabel('Dimension')
        axes[0].set_ylabel('r')
        axes[0].set_title('Average:' + str(np.nanmean(r)))

        best = np.argmax(r)
        axes[1].scatter(y_true[:, best], y_pred[:, best])
        axes[1].set_xlabel('True')
        axes[1].set_ylabel('Predicted')
        axes[1].set_title('Dimension:' + str(best))

        worst = np.argmin(r)
        axes[2].scatter(y_true[:, worst], y_pred[:, worst])
        axes[2].set_xlabel('True')
        axes[2].set_ylabel('Predicted')
        axes[2].set_title('Dimension:' + str(worst))

        plt.savefig(f'{save_dir}/bar-{prefix}emb_corr{suffix}.png',
                    bbox_inches='tight')

    return {
        f'{prefix}emb_corr_a': np.nanmean(r),
        f'{prefix}emb_corr_aT': np.nanmean(rT),
    }


def best_threshold(X, Y, T, best_x=0, best_y=1):
    '''Chooses the point a minimum distance to an ideal point. For ROC curves,
    that point is (0,1), for PR curves, it is (1,1)'''

    min_d, min_i = np.inf, 0
    for i, (x, y) in enumerate(zip(X, Y)):
        d = np.sqrt((best_x - x)**2 + (best_y - y)**2)
        if d < min_d:
            min_d, min_i = d, i

    return X[min_i], Y[min_i], T[min_i]


def evaluate_roc(predictions,
                 labels,
                 index2word,
                 train_freqs,
                 save_dir,
                 title='',
                 prefix='',
                 suffix=''):
    '''
        Evaluate ROC performance of the model.

        predictions: (N,n_classes) of activations
        labels: (N,n_classes) one-hot encoding
    '''

    rocs = {}
    lines = []
    scores = []
    train_word_freqs = []
    test_word_freqs = []
    fprs, tprs = [], []

    # Remove classes with no examples.
    target_classes = np.array(np.sum(labels, axis=0).nonzero())[0]
    labels = labels[:, target_classes]
    predictions = softmax(predictions[:, target_classes], axis=-1)
    i2w = {j: index2word[i] for j, i in enumerate(target_classes)}

    n_examples, n_classes = predictions.shape

    # Go over each class and compute AUC
    for i in range(n_classes):

        word = i2w[i]
        train_cnt = train_freqs[word]  # n_examples in training
        n_true = np.count_nonzero(labels[:, i])  # n_examples in dataset

        # Calculate ROC and AUC
        probs = predictions[:, i]
        c_labels = labels[:, i]
        fpr, tpr, thresh = roc_curve(c_labels, probs)
        score = auc(fpr, tpr)
        scores.append(score)
        rocs[word] = score

        # Save extra info for later
        fprs.append(fpr)
        tprs.append(tpr)
        train_word_freqs.append(train_cnt)
        test_word_freqs.append(n_true)

        # Compute confusion matrix for best threshold
        x, y, threshold = best_threshold(fpr, tpr, thresh)
        y_pred = probs >= threshold
        tn, fp, fn, tp = confusion_matrix(c_labels, y_pred).ravel()

        lines.append((word, n_true, train_cnt, score, tp, fp, fn, tn))

    # Compute weighted averages
    scores = np.array(scores)
    avg_auc = scores.mean()

    train_word_freqs = np.array(train_word_freqs)
    normed_freqs = train_word_freqs / train_word_freqs.sum()
    train_weighted_avg = (scores * normed_freqs).sum()

    test_word_freqs = np.array(test_word_freqs)
    normed_freqs = test_word_freqs / test_word_freqs.sum()
    test_weighted_avg = (scores * normed_freqs).sum()

    # skl_macro = roc_auc_score(labels, predictions, average='macro', multi_class='ovr')
    # skl_weighted = roc_auc_score(labels, predictions, average='weighted', multi_class='ovr')

    cols = 'word,freq_ds,freq_train,rocauc,tp,fp,fn,tn'.split(',')
    df = pd.DataFrame.from_records(lines, columns=cols)
    if suffix is not None:
        extras = {k: v for k, v in [s.split('_') for s in suffix.split('-')]}
        for key in extras:
            df[key] = extras[key]
        df.set_index(['word'] + list(extras.keys()))

    # Plot
    if save_dir is not None:
        N = scores.size
        # Plot histogram and AUC as a function of num of examples
        _, ax = plt.subplots(1, 1)
        ax.scatter(train_word_freqs, scores, marker='.')
        ax.set_xlabel('# examples')
        ax.set_ylabel('AUC')
        ax.set_title(f'{title} | avg: {test_weighted_avg:.3f} | N = {N}')
        ax.set_yticks(np.arange(0., 1.1, 0.1))
        ax.grid()
        plt.savefig(f'{save_dir}/rocauc_examples-{suffix}.png',
                    bbox_inches='tight')
        plt.close()

        # Plot a histogram of the AUCs
        _, ax = plt.subplots(1, 1)
        ax.hist(scores, bins=20)
        ax.set_xlabel('AUC')
        ax.set_ylabel('# labels')
        ax.set_title(f'{title} | avg: {test_weighted_avg:.3f} | N = {N}')
        ax.set_xticks(np.arange(0., 1., 0.1))
        plt.savefig(f'{save_dir}/rocauc_hist-{suffix}.png', bbox_inches='tight')
        plt.close()

        # Plot all curves on one plot
        _, ax = plt.subplots(1, 1)
        for fpr, tpr in zip(fprs, tprs):
            ax.plot(fpr, tpr, lw=1, alpha=0.8)
        ax.plot([0, 1], [0, 1], color='gray', linestyle='--')
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title(f'{title} | avg: {test_weighted_avg:.3f} | N = {N}')
        plt.savefig(f'{save_dir}/rocauc_all-{suffix}.png', bbox_inches='tight')
        plt.close()

    return {
        f'{prefix}rocauc': avg_auc,
        f'{prefix}rocauc_stddev': scores.std(),
        f'{prefix}rocauc_train_w_avg': train_weighted_avg,
        f'{prefix}rocauc_test_w_avg': test_weighted_avg,
        f'{prefix}rocauc_n': scores.size,
        f'{prefix}rocauc_df': df
    }


def evaluate_topk(predictions,
                  labels,
                  index2word,
                  train_freqs,
                  save_dir,
                  prefix='',
                  suffix=''):
    '''
        Evaluate top-k performance of the model. This assumes the activations
        can be interpreted as probabilities.

        TODO: update this documenation
        predictions: (N,n_classes) of activations
        labels: (N,n_classes) of labels
        i2w: mapping from index to word
        train_freqs: Counter object of training labels
    '''

    # Reduce classes to those in test set
    target_classes = np.array(np.sum(labels, axis=0).nonzero())[0]
    predictions = predictions[:, target_classes]
    labels = labels[:, target_classes]
    i2w = {j: index2word[i] for j, i in enumerate(target_classes)}

    ranks = []
    guesses = []
    class_freq = defaultdict(int)  # of times a class was in this set
    class_rank = defaultdict(list)  # rank per class
    class_n_pred = defaultdict(int)  # of times a class was predicted as top1
    class_n_correct = defaultdict(int)  # of times a class was correctly pred.
    n_examples, n_classes = predictions.shape

    # Go through each example and calculate its rank
    for i in range(n_examples):
        y_true_idx = labels[i].nonzero()[0][0]
        word = i2w[y_true_idx]

        instance_preds = np.argsort(predictions[i])[::-1]
        rank = np.where(y_true_idx == instance_preds)[0][0]
        ranks.append(rank)

        # Update class statistics
        class_freq[y_true_idx] += 1
        class_rank[y_true_idx].append(rank)
        class_n_pred[instance_preds[0]] += 1
        if rank == 0:
            class_n_correct[instance_preds[0]] += 1

        guesses.append([word, rank] + [i2w[j] for j in instance_preds[:10]])

    n_examples = len(ranks)
    ranks = np.array(ranks)
    top1 = sum(ranks == 0) / len(ranks)
    top5 = sum(ranks < 5) / len(ranks)
    top10 = sum(ranks < 10) / len(ranks)

    # Calculate chance levels based on training word frequencies
    labels_flat = np.where(labels == 1)[1]
    freqs = Counter([i2w[i] for i in labels_flat])
    freqs = np.array([freqs[w] for w, _ in train_freqs.most_common()])
    freqs = freqs[freqs != 0]
    chances = (freqs / freqs.sum()).cumsum()

    # Why would this happen?
    if len(chances) == 0:
        chances = [0] * 10
    if len(chances) < 10:
        chances = chances.tolist() + [0] * 10

    # Save top-10 predictions for each instance
    cols = ['word', 'rank'] + list(range(1, 11))
    df = pd.DataFrame.from_records(guesses, columns=cols)

    # Save class metrics
    class_accuracy = {i: class_n_correct[i] / class_freq[i]
                      for i in class_freq}
    records = [(i2w[i],
               np.mean(class_rank[i]),
               class_accuracy[i],
               train_freqs[i2w[i]],
               class_freq[i],
               class_n_pred[i],
               class_n_correct[i]) for i in range(n_classes)]
    cols = ['word', 'avg_rank', 'accuracy', 'freq_train',
            'freq_ds', 'n_predicted', 'n_correct']
    df2 = pd.DataFrame.from_records(records, columns=cols)

    if suffix is not None:
        extras = {k: v for k, v in [s.split('_') for s in suffix.split('-')]}
        for key in extras:
            df[key] = extras[key]
            df2[key] = extras[key]
        df2.set_index(['word'] + list(extras.keys()))

    return {
        f'{prefix}top1': top1,
        f'{prefix}top5': top5,
        f'{prefix}top10': top10,
        f'{prefix}top1_chance': chances[0],
        f'{prefix}top5_chance': chances[4],
        f'{prefix}top10_chance': chances[9],
        f'{prefix}n_classes': n_classes,
        f'{prefix}topk_guesses_df': df,
        f'{prefix}topk_df': df2,
    }


def class_to_vecs_tree(labels, vecs):
    """Create search trees for classes'vectors."""
    c_to_v = defaultdict(list)
    ret = {}
    ret_k = {}
    for i, v in enumerate(vecs):
        c_to_v[labels[i]].append(v)
    for c in c_to_v:
        ret_k[c] = len(c_to_v[c])
        ret[c] = cKDTree(c_to_v[c])
    return ret, ret_k


def get_class_predictions_kd(vecs_predictions,
                             t_vecs,
                             t_labels,
                             num_classes,
                             k=6):
    def _dsc(d):
        return -1.0 * d

    ctv_t, ctv_c = class_to_vecs_tree(t_labels, t_vecs)
    predictions = np.zeros((len(vecs_predictions), num_classes),
                           dtype=np.float64)

    for c in ctv_t:
        knn = min(k, ctv_c[c])
        dd, _ = ctv_t[c].query(vecs_predictions, knn)
        predictions[:,
                    c] = _dsc(np.average(dd, axis=1)) if knn > 1 else _dsc(dd)
    return predictions


def get_class_predictions(z_pred, z_true, y_true, n_classes, metric='cosine'):
    '''
    Transforms predicts embeddings to class scores - the average distance to
    all embeddings of a particular class becomes its score.

    z_pred - predicted embeddings
    z_true - actual embeddings
    y_true - actual embedding label values
    n_classes - number of classes in total
    '''
    predictions = np.zeros((len(z_pred), n_classes), dtype=np.float64)
    D = cdist(z_pred, z_true, metric=metric)
    for c in range(n_classes):
        predictions[:, c] = 1 - D[:, c == y_true].mean(axis=-1)
    return predictions


def evaluate_inclass_nn(ds_preds,
                        ds_words,
                        ds_embs,
                        all_emb,
                        all_words,
                        save_dir,
                        prefix='',
                        suffix=''):
    '''
    ds_preds: (N,D) dataset predicted embeddings
    ds_words: (N) dataset labels
    ds_embs: (N,D) true dataset embeddings
    all_emb: (M,D) all embeddings, including dataset
    all_words: (M) all labels
    '''

    rows = []
    ranks = []
    ranks_all = []

    # Compute (N,M) distance matrix
    D = cdist(ds_preds, all_emb, metric='cosine')
    DD = cdist(ds_embs, all_emb, metric='cosine')

    # Calculate top-k accuracy
    for i in range(D.shape[0]):
        word = ds_words[i]
        # select only in-class distances
        inclass_distances = D[i, (all_words == word)]
        true_index = DD[i, (all_words == word)].argmin()
        rank = (inclass_distances.argsort() == true_index).nonzero()[0].item()
        ranks.append(rank)

        true_index = DD[i].argmin()
        rank_all = (D[i].argsort() == true_index).nonzero()[0].item()
        ranks_all.append(rank_all)
        rows.append((word, inclass_distances.size, rank, rank == 0, rank_all))

    ranks = np.array(ranks)
    top1 = sum(ranks == 0) / len(ranks)
    top5 = sum(ranks < 5) / len(ranks)
    top10 = sum(ranks < 10) / len(ranks)

    ranks_all = np.array(ranks_all)
    top1_all = sum(ranks_all == 0) / len(ranks_all)
    top5_all = sum(ranks_all < 5) / len(ranks_all)
    top10_all = sum(ranks_all < 10) / len(ranks_all)

    # save csv
    cols = 'word,class_size,rank,hit,rank_all'.split(',')
    df = pd.DataFrame.from_records(rows, columns=cols)
    df.to_csv(save_dir + 'nn_inclass%s.csv' % suffix, index=False)

    return {
        prefix + 'nn_inclass_top1': top1,
        prefix + 'nn_inclass_top5': top5,
        prefix + 'nn_inclass_top10': top10,
        prefix + 'nn_all_top1': top1_all,
        prefix + 'nn_all_top5': top5_all,
        prefix + 'nn_all_top10': top10_all,
    }
