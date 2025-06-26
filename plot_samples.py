#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from qcnico.pixel2xyz import pxl2xyz
from qcnico.qcplots import plot_atoms_w_bonds
from qcnico.graph_tools import adjacency_matrix_sparse
from time import perf_counter



datadir = Path("~/Desktop/simulation_outputs/equivariant_MAC/samples").expanduser()
model_type = 'all_equiv'
nn = 60
Tsoftmax = 1.0
epoch = 1500

if model_type == 'vanilla_first':
    run_prefix = 'rot_90_conv_layers_60_nvanilla_' 
elif model_type == 'equivariant_first':
    run_prefix = 'rot_90_conv_layers_60_nequiv_'
elif model_type == 'all_equiv':
    run_prefix = 'rot_90_conv_layers_'



plot_type = 'pixel'

run_name = f'{run_prefix}{nn}'
if Tsoftmax:
    run_name += f'_Tsoftmax_{Tsoftmax}'
if epoch:
    run_name += f'_epoch-{epoch}'
# for n in [40,60]:
sample_npy = datadir / model_type / f"{run_name}.npy"
# sample_npy = datadir / f"rot_90_conv_layers_{n}_Tsoftmax_{T}.npy"
if sample_npy.exists():
    structures = np.load(sample_npy)
    for k,s in enumerate(structures):
        # if k > 0: break
        if plot_type == 'pixel':
            plt.imshow(s[0])
        else: # plot XYZ
            print('Starting pixel --> xyz conversion...', end = ' ')
            start = perf_counter()
            pos = pxl2xyz(s[0],pixel2angstroms=0.2)
            end = perf_counter()
            print(f'Done! [{end-start} seconds]')
            rCC = 1.8
            print('Building adjacency matrix..')
            M = adjacency_matrix_sparse(pos, rCC)
            fig, ax = plot_atoms_w_bonds(pos, M, show=False, dotsize=1.0,bond_lw=1.0)
        # plt.suptitle(f'{n} layers #{k} [$T = {T}$]')
        # plt.suptitle(f'{n} {layer_type} layers #{k}')
        plt.show()
else:
    print(f'file {sample_npy} not found')
