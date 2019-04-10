"""
Author: Srinivasan Sivanandan
"""
import sys
sys.path.append('datautils')
import argparse
import os
import tensorflow as tf
from datautils import dataset
from prediction import GRU, GRUM, GRUD, LSTM, LSTMM, LSTMSimple
import warnings
import pytest
warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.logging.set_verbosity(tf.logging.ERROR)

'''
args.batch_size=100
args.data_path="../../data/train"
args.model_path=None
args.result_path='results'
args.lr=0.001
args.epochs=100
args.n_inputs=36
args.n_hidden_units=100
args.n_classes=1
args.checkpoint_dir='checkpoint'
args.log_dir='logs'
args.normalize=True
args.dropout_rate=0.7
args.celltype='GRUD'
args.experiment='GRUDAct'
args.threshold=0.5

'''
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Arguments for sepsis prediction')
    parser.add_argument('--batch-size', type=int, default=100)
    parser.add_argument('--data-path', type=str, default="data/DAE_imputed")
    parser.add_argument('--model-path', type=str, default=None)
    parser.add_argument('--result-path', type=str, default='results')
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--n-inputs', type=int, default=36)
    parser.add_argument('--n-hidden-units', type=int, default=100)
    parser.add_argument('--n-classes', type=int, default=1)
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoint',
                        help='Directory name to save the checkpoints')
    parser.add_argument('--log-dir', type=str, default='logs',
                        help='Directory name to save training logs')
    parser.add_argument('--normalize',type=int,default=1)
    parser.add_argument('--dropout-rate',type=float,default=0.8)
    parser.add_argument('--celltype', type=str, default='GRUM')
    parser.add_argument('--experiment', type=str, default='GRUM')
    parser.add_argument('--threshold', type=float, default=0.5)
    parser.add_argument('--impute-forward', type=int, default=0)
    parser.add_argument('--calculate-delay', type=int, default=1)
    parser.add_argument('--imputation-method', type=str, default='mean')
    parser.add_argument('--early-stopping-patience', type=int, default=5)
    parser.add_argument('--seed', type=int, default=42)

    args = parser.parse_args()

    if args.normalize==0:
            args.normalize=False
    elif args.normalize==1:
            args.normalize=True

    if args.calculate_delay==0:
            args.calculate_delay=False
    elif args.calculate_delay==1:
            args.calculate_delay=True

    if args.imputation_method=='forward':
        args.impute_forward=True
    elif args.imputation_method=='mean':
        args.impute_forward=False
    else:
        # Open for unified imputation interface
        args.impute_forward=False

    checkdir=args.checkpoint_dir
    logdir=args.log_dir
    base=args.data_path
    data_paths=[]
    max_auc = 0.0

    args.checkpoint_dir=os.path.join(checkdir, args.experiment)
    args.log_dir=os.path.join(logdir,args.experiment, args.imputation_method, ('_').join(['seed',str(args.seed)]))

    print("Experiment : %s, Cell Type : %s,  Imputation : %s, Seed : %d"%(args.experiment,
                                                                        args.celltype,
                                                                        args.imputation_method,
                                                                        args.seed))
    #Max length across all datasets = 336.
    #Setting min maxLength=336 for traindata for now!!
    #TODO: Find max of max lengths across all datasets and use that for setting this maxLength
    dataset=dataset.Dataset(path=args.data_path,
                            batchSize=args.batch_size,
                            train_ratio=0.8,
                            normalize=args.normalize,
                            padding=True,
                            maxLength=336,
                            imputeForward=args.impute_forward,
                            calculateDelay=args.calculate_delay,
                            seed=args.seed)

    lrs=[0.001]
    for lr in lrs:
        args.lr=lr
        print("max epochs: %2d"%(args.epochs))
        tf.reset_default_graph()
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        with tf.Session(config=config) as sess:
            if args.celltype == "LSTM":
                model = LSTM.LSTM(sess,
                                  args=args,
                                  train_data=dataset.train_data,
                                  validation_data=dataset.val_data,
                                  test_data=dataset.test_data)
            elif args.celltype == "LSTMM":
                model = LSTMM.LSTMM(sess,
                                  args=args,
                                  train_data=dataset.train_data,
                                  validation_data=dataset.val_data,
                                  test_data=dataset.test_data)
            elif args.celltype == "LSTMSimple":
                model = LSTMSimple.LSTMSimple(sess,
                              args=args,
                              train_data=dataset.train_data,
                              validation_data=dataset.val_data,
                              test_data=dataset.test_data)
            elif args.celltype == "GRU":
                model = GRU.GRU(sess,
                                  args=args,
                                  train_data=dataset.train_data,
                                  validation_data=dataset.val_data,
                                  test_data=dataset.test_data)
            elif args.celltype == "GRUM":
                model = GRUM.GRUM(sess,
                                  args=args,
                                  train_data=dataset.train_data,
                                  validation_data=dataset.val_data,
                                  test_data=dataset.test_data)
            elif args.celltype == "GRUD":
                model = GRUD.GRUD(sess,
                                  args=args,
                                  train_data=dataset.train_data,
                                  validation_data=dataset.val_data,
                                  test_data=dataset.test_data)
            # build computational graph
            model.build()

            # Train model - (Internally validate the model on test set)
            max_auc, best_epoch = model.train()

            # Reproducing validation results from best epoch
            val_acc, val_auc, val_loss, val_tp, val_fp, val_tn, val_fn, val_counter = model.test(val=True, test_epoch=best_epoch, load_checkpoint=True)
            print(val_auc)
            # assert val_auc == pytest.approx(max_auc)

            # Test model and generate results for test data
            test_acc, test_auc, test_tp, test_fp, test_tn, test_fn, test_sens, test_spec = model.test(val=False, test_epoch=best_epoch, generate_files=True, load_checkpoint=True)

        result_file = open(os.path.join(args.result_path, args.experiment, args.imputation_method, ('_').join(['seed',str(args.seed)]), 'result.auc'),"w")
        result_file.write("val auc: {}".format(max_auc))
        result_file.write("\nval tp: {}, fp: {}, tn: {}, fn: {}".format(val_tp, val_fp, val_tn, val_fn))
        result_file.write("\ntest auc: {}".format(test_auc))
        result_file.write("\ntest tp: {}, fp: {}, tn: {}, fn: {}".format(test_tp, test_fp, test_tn, test_fn))
        result_file.write("\ntest sensitivity: {}, specificity: {}".format(test_sens, test_spec))
        result_file.close()

