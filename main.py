import utilities as utils
import os
import json

if __name__ == '__main__':
    data_root = 'datasets/unity-data'
    output_label_dir = 'datasets/labels/train'

    # utils.convert_annotations(data_root, output_label_dir)
    utils.separate_pngs(data_root, 'datasets/images/train')
    # os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
    # utils.train()

