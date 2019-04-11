from utilityscore import compute_scores_2019, load_column
import argparse
import os
import matplotlib.pyplot as plt
import numpy as np


RNN_MODELS = ['LSTM', 'LSTMM', 'LSTMSimple', 'GRU', 'GRUM', 'GRUD']
METRICS = ['Utility', 'Sensitivity', 'Specificity']


def generate_plots(args):
    rnn = []
    non_rnn = []

    for model in os.listdir(args.metrics_path):
        model_path = os.path.join(args.metrics_path, model)
        plot_roc(model, model_path, args.plot_path)
        if model in RNN_MODELS:
            rnn.append(model)
        else:
            non_rnn.append(model)

    for metric in METRICS:
        if rnn:
            plot_metric(args.metrics_path, rnn, metric, 'RNN', args.plot_path)
        if non_rnn:
            plot_metric(args.metrics_path, non_rnn, metric, 'Non-RNN', args.plot_path)


def plot_roc(model, model_path, plot_path):
    plt.style.use('ggplot')
    plt.figure(dpi=150)
    plt.plot([0, 1], [0, 1], 'k--')

    # colors = ['m', 'c', 'g', 'b']
    count=0

    for imputation_method in os.listdir(model_path):
        imputation_dir = os.path.join(model_path, imputation_method)

        fpr = load_column(os.path.join(imputation_dir, 'seed_1.roc'), 'FPR')
        tpr = load_column(os.path.join(imputation_dir, 'seed_1.roc'), 'TPR')
        auc = load_column(os.path.join(imputation_dir, 'seed_1.metrics'), 'AUROC')

        # plt.plot(fpr, tpr, label='{} {}'.format(imputation_method, '{0:.3f}'.format(auc[0])), color=colors[count])
        plt.plot(fpr, tpr, label='{} {}'.format(imputation_method, '{0:.3f}'.format(auc[0])))
        count += 1
    plt.legend()
    plt.title('{} Roc Curve'.format(model))

    roc_dir = '{}/roc'.format(plot_path)

    if not os.path.exists(roc_dir):
        os.makedirs(roc_dir)

    plt.savefig('{}/{}.png'.format(roc_dir, model))

    plt.close()


def plot_metric(metrics_path, models, metric, type, plot_path):
    d = {}
    for model in os.listdir(metrics_path):
        if model in models:
            model_dir = os.path.join(metrics_path, model)
            for imputation_method in os.listdir(model_dir):
                if imputation_method not in d:
                    d[imputation_method] = {}
                imputation_dir = os.path.join(model_dir, imputation_method)
                all_seed_metrics = []
                for seed in os.listdir(imputation_dir):
                    if seed.endswith('.metrics'):
                        all_seed_metrics.append(load_column(os.path.join(imputation_dir, seed), metric)[0])

                if 'means' not in d[imputation_method]:
                    d[imputation_method]['means'] = []
                if 'stds' not in d[imputation_method]:
                    d[imputation_method]['stds'] = []

                mean = np.mean(all_seed_metrics)
                std = np.std(all_seed_metrics)

                d[imputation_method]['means'].append(mean)
                d[imputation_method]['stds'].append(std)

    plt.style.use('ggplot')
    plt.figure(dpi=150)
    N = len(models)
    fig, ax = plt.subplots()
    ind = np.arange(N)
    width = 0.35
    rects = []
    rects_0 = []
    count = 0
    # colors = ['m', 'c', 'g', 'b']
    imputation_types = []

    for imputation in d:
        imputation_types.append(imputation)
        means = d[imputation]['means']
        stds = d[imputation]['stds']
        rect = ax.bar(ind + count * width, means, width, yerr=stds)
        # rect = ax.bar(ind + count*width, means, width, color=colors[count], yerr=stds)
        rects.append(rect)
        rects_0.append(rect[0])
        count += 1

    ax.set_ylabel(metric)
    ax.set_title('{} Methods - {}'.format(type, metric))
    ax.set_xticks(ind + width/2)
    ax.set_xticklabels(models)
    ax.legend(rects_0, imputation_types)

    metric_dir = '{}/metrics'.format(plot_path)

    if not os.path.exists(metric_dir):
        os.makedirs(metric_dir)

    plt.savefig('{}/{}-{}.png'.format(metric_dir, type, metric))

    plt.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Arguments for sepsis prediction')
    parser.add_argument('--metrics_path', type=str, default='metrics')
    parser.add_argument('--plot_path', type=str, default='plots')

    args = parser.parse_args()

    generate_plots(args)
