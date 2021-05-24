import argparse
import os
import paddle
import paddle.optimizer
import paddle.io
from repmlp_resnet import *
from repmlp import repmlp_model_convert
parser = argparse.ArgumentParser(description='RepVGG Conversion')
parser.add_argument('load', metavar='LOAD', help='path to the weights file')
parser.add_argument('save', metavar='SAVE', help='path to the weights file')
parser.add_argument('-a', '--arch', metavar='ARCH', default=\
    'RepMLP-Res50-light-224')
def convert():
    args = parser.parse_args()
    if args.arch == 'RepMLP-Res50-light-224':
        train_model = create_RepMLPRes50_Light_224(deploy=False)
    else:
        raise ValueError('TODO')
    if os.path.isfile(args.load):
        print("=> loading checkpoint '{}'".format(args.load))
        checkpoint = paddle.load(args.load)
        if 'state_dict' in checkpoint:
            checkpoint = checkpoint['state_dict']
        elif 'model' in checkpoint:
            checkpoint = checkpoint['model']
        ckpt = {k.replace('module.', ''): v for k, v in checkpoint.items()}
        print(ckpt.keys())
        train_model.load_state_dict(ckpt)
    else:
        print("=> no checkpoint found at '{}'".format(args.load))
    repmlp_model_convert(train_model, save_path=args.save)


if __name__ == '__main__':
    convert()
