"""
Author: Srinivasan Sivanandan
"""
import argparse
import os
import tensorflow as tf
from datautils import imputerDataset
from imputation import DAEImpute
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
    parser.add_argument('--data-path', type=str, default="../../data/train")
    parser.add_argument('--model-path', type=str, default=None)
    parser.add_argument('--result-path', type=str, default='results')
    parser.add_argument('--lr', type=float, default=0.001)
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
    parser.add_argument('--celltype', type=str, default='GRU')
    parser.add_argument('--experiment', type=str, default='DAE')
    parser.add_argument('--threshold', type=float, default=0.5)
    parser.add_argument('--early-stopping-patience', type=int, default=5)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--missing-rate', type=float, default=0.4)
    parser.add_argument('--min-mask-epoch', type=int, default=10)

    args = parser.parse_args()

    if args.normalize==0:
            args.normalize=False
    if args.normalize==1:
            args.normalize=True

    checkdir=args.checkpoint_dir
    logdir=args.log_dir
    base=args.data_path
    data_paths=[]
    max_auc = 0.0

    args.checkpoint_dir=os.path.join(checkdir, args.experiment)
    args.log_dir=os.path.join(logdir,args.experiment)

    print("Imputation Experiment : %s, Cell Type : %s, Seed : %d"%(args.experiment, 
                                                                        args.celltype,
                                                                        args.seed))

    dataset=imputerDataset.Dataset(path=args.data_path,
                                        batchSize=args.batch_size,
                                        train_ratio=0.8, 
                                        normalize=True, 
                                        padding=True,
                                        seed = args.seed,
                                        missingRate=args.missing_rate,
                                        minMaskEpoch=args.min_mask_epoch)
        
    lrs=[0.001]
    for lr in lrs:
        args.lr=lr
        print("epoch: %2d"%(args.epochs))
        tf.reset_default_graph()
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        with tf.Session(config=config) as sess:
            model = DAEImpute.DAE(sess,
                            args=args,
                            train_data=dataset.train_data,
                            val_data=dataset.val_data,
                            test_data=dataset.val_data)

            # build computational graph
            model.build()

            # Train model - (Internally validate the model on test set)
            min_loss, best_epoch = model.train()

            # Reproducing validation results from best epoch            
            val_loss = model.test(val=True, test_epoch=best_epoch, load_checkpoint=True)
            assert val_loss == pytest.approx(min_loss)
            
            # Test model and generate results for test data
            test_loss = model.test(val=False, test_epoch=best_epoch, generate_files=True, load_checkpoint=True)

        print("min mse is: " + str(min_loss))
        result_file = open(os.path.join(args.result_path, args.experiment, args.imputation_method, ('_').join(['seed',str(args.seed)]), 'result.mse'),"w")
        result_file.write("val mse: {}".format(min_mse))
        result_file.write("\ntest mse: {}".format(min_mse))
        result_file.close()