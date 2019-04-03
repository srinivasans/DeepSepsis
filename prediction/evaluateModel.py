import sys
sys.path.append("..")
from datautils import readData
import tensorflow as tf
import GRUD
import os
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Arguments for sepsis prediction')
    parser.add_argument('--batch-size', type=int, default=100)
    parser.add_argument('--data-path', type=str, default="../../data/train")
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
    parser.add_argument('--experiment', type=str, default='GRU_WeightedU_0.025')
    parser.add_argument('--threshold', type=float, default=0.5)

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
    
    train_data=readData.ReadData(path=args.data_path,
                                batchSize=args.batch_size, 
                                isTrain=True, 
                                normalize=args.normalize, 
                                padding=True, 
                                mean = None, 
                                std = None)
    
    test_data=readData.ReadData(path=args.data_path.replace("train","test"),
                            batchSize=args.batch_size,
                            isTrain=False,
                            normalize=args.normalize,
                            padding=True,
                            mean=train_data.mean,
                            std=train_data.std,
                            maxLength=train_data.maxLength)

    tf.reset_default_graph()
    config = tf.ConfigProto() 
    config.gpu_options.allow_growth = True 
    with tf.Session(config=config) as sess:
        model = GRUD.grud(sess,
                    args=args,
                    train_data=train_data,
                    test_data=test_data
                    )
        model.build()
        acc, auc, loss= model.test(checkpoint_dir='checkpoint/GRU_WeightedU_0.025/GRU_WeightedU_0.025_0.001_100_True/epoch400', test_epoch=400, generate_files=True)
        print(auc)