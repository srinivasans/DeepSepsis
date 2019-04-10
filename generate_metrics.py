from utilityscore import compute_scores_2019, load_column
import argparse
import os


def compute_metrics(args):
    label_directory = args.test_data_path
    dataDir = os.path.join(args.result_path, args.experiment)

    for imputation in os.listdir(dataDir):
        imputation_directory = os.path.join(dataDir, imputation)
        for seed in os.listdir(imputation_directory):
            prediction_directory = os.path.join(imputation_directory, seed)
            auroc, auprc, accuracy, f_measure, normalized_observed_utility, thresholds, tpr, fpr, tn, fp, fn, tp = compute_scores_2019(label_directory, prediction_directory)
            save_metrics_dir = os.path.join(args.metrics_path, args.experiment, imputation)
            if not os.path.exists(save_metrics_dir):
                os.makedirs(save_metrics_dir)

            metrics_string = 'AUROC|AUPRC|Accuracy|F-measure|Utility|TN|FP|FN|TP|Sensitivity|Specificity\n{}|{}|{}|{}|{}|{}|{}|{}|{}|{}|{}'.format(auroc, auprc, accuracy, f_measure, normalized_observed_utility, tn, fp, fn, tp, tp/(tp+fn), tn/(tn+fp))
            metrics_file = os.path.join(save_metrics_dir, '{}.metrics'.format(seed))
            with open(metrics_file, 'w') as f:
                f.write(metrics_string)

            roc_headers = 'Thresholds|TPR|FPR\n'
            roc_file = os.path.join(save_metrics_dir, '{}.roc'.format(seed))
            with open(roc_file, 'w') as f:
                f.write(roc_headers)
                for i in range(len(thresholds)):
                    f.write('{}|{}|{}\n'.format(thresholds[i], tpr[i], fpr[i]))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Arguments for sepsis prediction')
    parser.add_argument('--experiment', type=str, default='LSTMM')
    parser.add_argument('--result_path', type=str, default='results') # prediction path
    parser.add_argument('--metrics_path', type=str, default='metrics')
    parser.add_argument('--test_data_path', type=str, default='data/challenge_test_data') # ground truth path

    args = parser.parse_args()

    compute_metrics(args)
