#!/usr/bin/env python

import argparse
from args_utils import add_training_args



parser = argparse.ArgumentParser()

# sample generation parameters
parser.add_argument('--training_run_name', type=str)
parser.add_argument('--experiment_name', type=str, default='')
parser.add_argument('--epoch_chkpt', type=int, default=0) # epoch index from which the model should be loaded (make sure a checkpoint file corresponding to that epoch exists); if set to negative, find the checkpoint with lowest test loss
parser.add_argument('--bound_type', type = str, default = 'empty') # what is outside the image during training and generation 'empty'
parser.add_argument('--boundary_layers', type = int, default = 0) # number of layers of conv_field between sample and actual image boundary
parser.add_argument('--sample_outpaint_ratio', type = int, default = 7) # size of sample images, relative to the input images
parser.add_argument('--softmax_temp', type = float, default = 1.0)
parser.add_argument('--sample_generation_mode', type = str, default = 'parallel') # 'parallel' or 'serial' - serial is currently untested
parser.add_argument('--sample_batch_size', type = int, default = 1000) # maximum sample batch size - no automated test but can generally be rather large (1e3),
parser.add_argument('--generation_period', type = int, default = 1000) # how often to run (expensive) generation during training
# utility of higher batch sizes for parallel generation is only realized with extremely large samples
parser.add_argument('--n_samples', type = int, default = 1) # number of samples to generate


configs,unknown= parser.parse_known_args()
configs = add_training_args(configs)

for k, v in vars(configs).items():
    print(f'{k}: {v}')