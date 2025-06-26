#!/usr/bin/env python
import os
import pickle
from utils import build_dataset
import argparse
import numpy as np


def get_2d_coords(coords_arr):
    return coords_arr[:,[0,2]]
    


parser = argparse.ArgumentParser()
parser.add_argument('--training_dataset', type=str, default='amorphous')
parser.add_argument('--dataset_seed', type = int, default = 0)
parser.add_argument('--sample_outpaint_ratio', type = int, default = 5)
parser.add_argument('--vanilla_layers', type = int, default=1)
parser.add_argument('--equivariant_layers', type = int, default=1)
parser.add_argument('--conv_size', type = int, default = 3) # ODD NUMBER
parser.add_argument('--experiment_name', type=str, default='check_amorph_tr_strucs')

configs = parser.parse_args()

outdir = configs.experiment_name
if not os.path.isdir(outdir):
    os.makedirs(outdir,exist_ok=True)

dataset = build_dataset(configs)



samples = dataset.samples
nsamples = samples.shape[0]


seed = 64
rng = np.random.default_rng(64)
isamples_p6dot7 = np.load('data/p6dot7andmore.npy', allow_pickle=True).astype('int')
nsamples_plot = 10
isamples_plot = rng.integers(isamples_p6dot7.shape[0], size=nsamples_plot)

pixel_sample = samples[isamples_plot]
np.save(f"{outdir}/check_amorphous_sample.py", pixel_sample)

all_real_space = np.load('data/ac2d_coords_Meta.npy', allow_pickle=True)

real_space_sample = []
for i in isamples_p6dot7[isamples_plot]:
    real_space_sample.append(get_2d_coords(all_real_space[i]))

with open(f'{outdir}/check_amorphous_realspace.pkl', 'wb') as fo:
    pickle.dump(real_space_sample, fo)