import sys
sys.path.append("..")
from datautils import dataset
import tensorflow as tf
from prediction import GRUD
import os
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Arguments for sepsis prediction')
    parser.add_argument('--batch-size', type=int, default=100)
    parser.add_argument('--data-path', type=str, default="data/sepsis_data")
    parser.add_argument('--model-path', type=str, default=None)
    parser.add_argument('--result-path', type=str, default='results')
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--n-inputs', type=int, default=36)
    parser.add_argument('--n-hidden-units', type=int, default=72)
    parser.add_argument('--n-classes', type=int, default=1)
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoint',
                        help='Directory name to save the checkpoints')
    parser.add_argument('--log-dir', type=str, default='logs',
                        help='Directory name to save training logs')
    parser.add_argument('--normalize',type=int,default=1)
    parser.add_argument('--dropout-rate',type=float,default=0.5)
    parser.add_argument('--celltype', type=str, default='GRUD')
    parser.add_argument('--experiment', type=str, default='GRUM')
    parser.add_argument('--threshold', type=float, default=0.5)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--impute-forward', type=float, default=0)
    parser.add_argument('--imputation_method', type=str, default='mean')

    args = parser.parse_args()
    
    if args.normalize==0:
            args.normalize=False
    if args.normalize==1:
            args.normalize=True

    if args.impute_forward==0:
            args.impute_forward=False
    if args.impute_forward==1:
            args.impute_forward=True

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
                            seed=args.seed)

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
        model.build()
        test_acc, test_auc, test_tp, test_fp, test_tn, test_fn, sens, spec = model.test(checkpoint_dir='checkpoint/GRUM_C_Mean_40k/GRUM_C_Mean_40k_0.001_100_True/epoch5', 
                                                                                test_epoch=5, 
                                                                                generate_files=True, 
                                                                                val=False,
                                                                                load_checkpoint=True)
        print((test_acc, test_auc, test_tp, test_fp, test_tn, test_fn, sens, spec))