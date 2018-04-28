""" Process images with trained model """

from __future__ import print_function

import argparse
import os

from utilities.input_correction import Correction

__author__ = 'Cong Bao'

CHECKPOINT_PATH = './checkpoints/'
OUTPUT_PATH = './results/'

BATCH_SIZE = 128

CHECKPOINT_NAME = 'checkpoint.best.hdf5'

def main():
    """ parse parameters from command line and start processing images """
    parser = argparse.ArgumentParser()
    add_arg = parser.add_argument
    add_arg('-i', dest='input',  type=str, required=True,            help='directory of images')
    add_arg('-s', dest='shape',  type=int, required=True, nargs='+', help='width, height, channel of image')
    add_arg('-o', dest='output', type=str, default=OUTPUT_PATH,      help='directory to store processed images, default %s' % OUTPUT_PATH)
    add_arg('-b', dest='batch',  type=int, default=BATCH_SIZE,       help='batch size, default %s' % BATCH_SIZE)
    add_arg('--checkpoint-path', dest='path', type=str, default=CHECKPOINT_PATH, help='path to save checkpoint files, default %s' % CHECKPOINT_PATH)
    add_arg('--checkpoint-name', dest='name', type=str, default=CHECKPOINT_NAME, help='the name of checkpoint file, default %s' % CHECKPOINT_NAME)
    add_arg('--cpu-only',        dest='cpu',  action='store_true',               help='whether use cpu only or not, default False')
    args = parser.parse_args()
    if args.cpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
    corr = Correction().correct
    params = {
        'img_shape':       tuple(args.shape),
        'img_dir':         corr(args.input),
        'res_dir':         corr(args.output),
        'checkpoint_path': corr(args.path),
        'checkpoint_name': args.name,
        'batch_size':      args.batch
    }
    assert os.path.exists(params['checkpoint_path'])
    print('Image directory: %s'  % params['img_dir'])
    print('Output directory: %s' % params['res_dir'])
    print('Checkpoint path: %s'  % params['checkpoint_path'])
    print('Checkpoint name: %s'  % params['checkpoint_name'])
    print('Shape of image: %s'   %(params['img_shape'],))
    print('Batch size: %s'       % params['batch_size'])
    print('Running on %s' % ('CPU' if args.cpu else 'GPU'))
    if not os.path.exists(params['res_dir']):
        os.makedirs(params['res_dir'])
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    from enhancer import Enhancer
    enhancer = Enhancer(**params)
    enhancer.load_data(process=True)
    enhancer.load_model()
    try:
        enhancer.process()
    except (KeyboardInterrupt, SystemExit):
        print('Abort!')

if __name__ == '__main__':
    main()
