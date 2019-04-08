"""
Author: Srinivasan Sivanandan
"""
import sys
sys.path.append('datautils')
import argparse
import os
import tensorflow as tf
from datautils import dataset
from prediction import GRUD, LSTM


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
    parser.add_argument('--data-path', type=str, default="data/sepsis_data")
    parser.add_argument('--model-path', type=str, default=None)
    parser.add_argument('--result-path', type=str, default='results')
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--n-inputs', type=int, default=36)
    parser.add_argument('--n-hidden-units', type=int, default=72)
    parser.add_argument('--n-classes', type=int, default=1)
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoint',
                        help='Directory name to save the checkpoints')
    parser.add_argument('--log-dir', type=str, default='logs',
                        help='Directory name to save training logs')
    parser.add_argument('--normalize',type=int,default=1)
    parser.add_argument('--dropout-rate',type=float,default=0.8)
    parser.add_argument('--celltype', type=str, default='GRUD')
    parser.add_argument('--experiment', type=str, default='GRUD')
    parser.add_argument('--threshold', type=float, default=0.5)
    parser.add_argument('--impute-forward', type=int, default=0)
    parser.add_argument('--calculate-delay', type=int, default=1)
    parser.add_argument('--imputation_method', type=str, default='mean')
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
    args.log_dir=os.path.join(logdir,args.experiment)

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
                            calculateDelay=args.calculate_delay)

    lrs=[0.001]
    for lr in lrs:
        args.lr=lr
        print("epoch: %2d"%(args.epochs))
        tf.reset_default_graph()
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        with tf.Session(config=config) as sess:
            if args.celltype == "LSTM":
                model = LSTM.lstm(sess,
                                  args=args,
                                  train_data=dataset.train_data,
                                  validation_data=dataset.val_data,
                                  test_data=dataset.test_data)
            else:
                model = GRUD.grud(sess,
                                  args=args,
                                  train_data=dataset.train_data,
                                  validation_data=dataset.val_data,
                                  test_data=dataset.test_data)

            # build computational graph
            model.build()

            # Train model - (Internally validate the model on test set)
            auc, tp, fp, tn, fn = model.train()
            if auc > max_auc:
                max_auc = auc

            test_acc, test_auc, test_tp, test_fp, test_tn, test_fn, sens, spec = model.test(val=False)

    print("max auc is: " + str(max_auc))
    f2 = open(('_').join(["max_auc",args.experiment]),"w")
    f2.write("val auc: {}".format(max_auc))
    f2.write("\nval tp: {}, fp: {}, tn: {}, fn: {}".format(tp, fp, tn, fn))
    f2.write("\ntest auc: {}".format(test_auc))
    f2.write("\ntest tp: {}, fp: {}, tn: {}, fn: {}".format(test_tp, test_fp, test_tn, test_fn))
    f2.write("\ntest sensitivity: {}, specificity: {}".format(sens, spec))
    f2.close()

