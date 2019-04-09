import sys
sys.path.append("..")
from datautils import readDataImputation
import tensorflow as tf
import DAEImpute
import os
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Arguments for sepsis prediction')
    parser.add_argument('--batch-size', type=int, default=100)
    parser.add_argument('--data-path', type=str, default="../../data/train")
    parser.add_argument('--model-path', type=str, default=None)
    parser.add_argument('--result-path', type=str, default='results')
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--epochs', type=int, default=1000)
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
    parser.add_argument('--experiment', type=str, default='GRUD')
    parser.add_argument('--threshold', type=float, default=0.5)
    parser.add_argument('--early-stopping-patience', type=int, default=5)
    parser.add_argument('--seed', type=int, default=42)

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
                                        padding=True)
    tf.reset_default_graph()
    config = tf.ConfigProto() 
    config.gpu_options.allow_growth = True 
    with tf.Session(config=config) as sess:
        model = DAEImpute.DAE(sess,
                    args=args,
                    train_data=dataset.train_data,
                    val_data=dataset.val_data
                    test_data=dataset.test_data
                    )
        model.build()
        loss= model.test(checkpoint_dir='checkpoint/GRUD/GRUD_0.001_100_True/epoch70', test_epoch=70, generate_files=True)
        print(loss)