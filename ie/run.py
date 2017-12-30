""" Run the enhancer """

import argparse
import os

from utilities.input_correction import Correction

__author__ = 'Cong Bao'

GRAPH_PATH = './graphs/'
CHECKPOINT_PATH = './checkpoints/'
EXAMPLE_PATH = './examples/'

MODEL_TYPES = ['denoise', 'augment']
CORRUPT_TYPES = ['GS', 'MN', 'SP', 'ZIP']

MODEL_TYPE = 'denoise'
LEARNING_RATE = 0.001
BATCH_SIZE = 128
EPOCH = 50
CORRUPT_TYPE = 'GS'
CORRUPT_RATIO = 0.05

def main():
    """ parse parameters from command line and start the training of model """
    parser = argparse.ArgumentParser()
    add_arg = parser.add_argument
    add_arg('-m', dest='model',  type=str,   required=True,            help='the model to train, within %s' % MODEL_TYPES)
    add_arg('-i', dest='input',  type=str,   required=True,            help='directory of source images')
    add_arg('-s', dest='shape',  type=int,   required=True, nargs='+', help='width, height, channel of image')
    add_arg('-r', dest='rate',   type=float, default=LEARNING_RATE,    help='learning rate, default %s' % LEARNING_RATE)
    add_arg('-b', dest='batch',  type=int,   default=BATCH_SIZE,       help='batch size, default %s' % BATCH_SIZE)
    add_arg('-e', dest='epoch',  type=int,   default=EPOCH,            help='number of epoches, default %s' % EPOCH)
    add_arg('-T', dest='type',   type=str,   default=CORRUPT_TYPE,     help='type of corruption, within %s' % CORRUPT_TYPES)
    add_arg('-R', dest='ratio',  type=float, default=CORRUPT_RATIO,    help='ratio of corruption, default %s' % CORRUPT_RATIO)
    add_arg('--graph-path',      dest='graph',      type=str, default=GRAPH_PATH,      help='path to save tensor graphs, default %s' % GRAPH_PATH)
    add_arg('--checkpoint-path', dest='checkpoint', type=str, default=CHECKPOINT_PATH, help='path to save checkpoint files, default %s' % CHECKPOINT_PATH)
    add_arg('--example-path',    dest='example',    type=str, default=EXAMPLE_PATH,    help='path to save example images, default %s' % EXAMPLE_PATH)
    add_arg('--cpu-only',        dest='cpu',        action='store_true',               help='whether use cpu only or not, default False')
    args = parser.parse_args()
    if args.cpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
    assert args.model in MODEL_TYPES
    assert args.type in CORRUPT_TYPES
    corr = Correction().correct
    params = {
        'model_type':      args.model,
        'img_shape':       tuple(args.shape),
        'img_dir':         corr(args.input),
        'graph_path':      corr(args.graph),
        'checkpoint_path': corr(args.checkpoint),
        'example_path':    corr(args.example),
        'learning_rate':   args.rate,
        'batch_size':      args.batch,
        'epoch':           args.epoch,
        'corrupt_type':    args.type,
        'corrupt_ratio':   args.ratio
    }
    print('Model to train: %s'   % params['model_type'])
    print('Image directory: %s'  % params['img_dir'])
    print('Graph path: %s'       % params['graph_path'])
    print('Checkpoint path: %s'  % params['checkpoint_path'])
    print('Example path: %s'     % params['example_path'])
    print('Shape of image: %s'   %(params['img_shape'],))
    print('Learning rate: %s'    % params['learning_rate'])
    print('Batch size: %s'       % params['batch_size'])
    print('Epoches to train: %s' % params['epoch'])
    print('Corruption type: %s'  % params['corrupt_type'])
    print('Corruption ratio: %s' % params['corrupt_ratio'])
    print('Running on %s' % ('CPU' if args.cpu else 'GPU'))
    if not os.path.exists(params['checkpoint_path']):
        os.makedirs(params['checkpoint_path'])
    if not os.path.exists(params['example_path']):
        os.makedirs(params['example_path'])
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    from enhancer import Enhancer
    enhancer = Enhancer(**params)
    enhancer.load_data()
    enhancer.build_model()
    enhancer.train_model()

if __name__ == '__main__':
    main()
