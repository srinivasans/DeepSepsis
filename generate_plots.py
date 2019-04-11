from utilityscore import compute_scores_2019, load_column
import argparse
import os
import matplotlib.pyplot as plt
import numpy as np


RNN_MODELS = ['LSTM', 'LSTMM', 'LSTMSimple', 'GRU', 'GRUM', 'GRUD']
NON_RNN_MODELS = ['COX', 'RF', 'XG', 'RLR']
METRICS = ['Utility', 'Sensitivity', 'Specificity']


MODEL_NAME_MAPPTING = {'COX': 'Cox PH', 'RF': 'RF', 'XG': 'XGBoost', 'RLR': 'RLR',
                       'LSTMSimple': 'LSTM-MD', 'LSTM': 'LSTM', 'LSTMM': 'LSTM-M',
                       'GRU': 'GRU', 'GRUM': 'GRU-M', 'GRUD': 'GRU-Decay'}

RNN_IMPUTATION = ['mean', 'forward', 'grud', 'dae']
NON_RNN_IMPUTATION = ['mean', 'forward']

def generate_plots(args):
    rnn = []
    non_rnn = []

    for model in os.listdir(args.metrics_path):
        model_path = os.path.join(args.metrics_path, model)
        if model in RNN_MODELS:
            plot_roc(model, model_path, args.plot_path, 'RNN')
        else:
            plot_roc(model, model_path, args.plot_path, 'Non-RNN')
        if model in RNN_MODELS:
            rnn.append(model)
        elif model in NON_RNN_MODELS:
            non_rnn.append(model)

    for metric in METRICS:
        if rnn:
            plot_metric(args.metrics_path, rnn, metric, 'RNN', args.plot_path)
        if non_rnn:
            plot_metric(args.metrics_path, non_rnn, metric, 'Non-RNN', args.plot_path)


def plot_roc(model, model_path, plot_path, type):
    plt.style.use('ggplot')
    plt.figure(figsize=[4, 4], dpi=150)
    plt.plot([0, 1], [0, 1], 'k--')

    print("Metric: AUC")

    # colors = ['m', 'c', 'g', 'b']
    count=0

    if type == 'RNN':
        imputation_lst = RNN_IMPUTATION
    else:
        imputation_lst = NON_RNN_IMPUTATION

    for imputation_method in imputation_lst:
        imputation_dir = os.path.join(model_path, imputation_method)

        fpr = load_column(os.path.join(imputation_dir, 'seed_1.roc'), 'FPR')
        tpr = load_column(os.path.join(imputation_dir, 'seed_1.roc'), 'TPR')
        auc = load_column(os.path.join(imputation_dir, 'seed_1.metrics'), 'AUROC')

        # plt.plot(fpr, tpr, label='{} {}'.format(imputation_method, '{0:.3f}'.format(auc[0])), color=colors[count])
        plt.plot(fpr, tpr, label='{} {}'.format(imputation_method, '{0:.3f}'.format(auc[0])), alpha=0.7)
        aucs = []
        for seed in os.listdir(imputation_dir):
            if seed.endswith('.metrics'):
                aucs.append(load_column(os.path.join(imputation_dir, seed), 'AUROC'))
        auc_mean = np.mean(aucs)
        auc_std = np.std(aucs)

        print('Model: {} Imputation Type: {} Avg: {} Std: {}'.format(model, imputation_method, auc_mean, auc_std))
        count += 1
    plt.legend()
    plt.title('{} ROC Curve'.format(MODEL_NAME_MAPPTING[model]))

    roc_dir = '{}/roc'.format(plot_path)

    if not os.path.exists(roc_dir):
        os.makedirs(roc_dir)

    plt.show()

    plt.savefig('{}/{}.png'.format(roc_dir, model))

    plt.close()


def plot_metric(metrics_path, models, metric, type, plot_path):

    print("Metric: {}".format(metric))

    model_names = []
    d = {}

    if type == 'RNN':
        imputation_lst = RNN_IMPUTATION
    else:
        imputation_lst = NON_RNN_IMPUTATION

    for model in os.listdir(metrics_path):
        if model in models:
            model_dir = os.path.join(metrics_path, model)
            model_names.append(MODEL_NAME_MAPPTING[model])
            for imputation_method in imputation_lst:
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


                print('Model: {} Imputation Type: {} Avg: {} Std: {}'.format(model, imputation_method, mean, std))

    plt.style.use('ggplot')


    plt.figure(dpi=150)
    N = len(models)
    fig, ax = plt.subplots()
    ind = np.arange(N)
    width = 0.2
    rects = []
    rects_0 = []
    count = 0
    # colors = ['m', 'c', 'g', 'b']
    imputation_types = []

    for imputation in imputation_lst:
        means = d[imputation]['means']
        stds = d[imputation]['stds']
        rect = ax.bar(ind + count * width, means, width, yerr=stds)
        # rect = ax.bar(ind + count*width, means, width, color=colors[count], yerr=stds)
        rects.append(rect)
        rects_0.append(rect[0])
        count += 1

    if type == 'RNN':
        factor = 1.5
    else:
        factor = 0.5

    ax.set_ylabel(metric)
    ax.set_title('{} Methods - {}'.format(type, metric))
    ax.set_xticks(ind + factor*width)
    ax.set_xticklabels(model_names)
    ax.legend(rects_0, imputation_lst)

    metric_dir = '{}/metrics'.format(plot_path)

    if not os.path.exists(metric_dir):
        os.makedirs(metric_dir)

    plt.ylim(0, 1)
    plt.show()

    plt.savefig('{}/{}-{}.png'.format(metric_dir, type, metric))

    plt.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Arguments for sepsis prediction')
    parser.add_argument('--metrics_path', type=str, default='metrics')
    parser.add_argument('--plot_path', type=str, default='plots')

    args = parser.parse_args()

    generate_plots(args)
