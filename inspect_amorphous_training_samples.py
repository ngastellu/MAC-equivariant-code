#!/usr/bin/env python

import pickle
import numpy as np
import matplotlib.pyplot as plt
from qcnico.qcplots import plot_atoms


pixel_imgs = np.load('check_amorph_tr_strucs/check_amorphous_pxl.npy')

with open('check_amorph_tr_strucs/check_amorphous_realspace.pkl', 'rb') as fo:
    real_space_pos = pickle.load(fo)

for pxl, rsp in zip(pixel_imgs, real_space_pos):
    fig, axs = plt.subplots(1,2)
    axs[0].imshow(pxl[0], origin='lower')

    fig, axs[1] = plot_atoms(rsp, plt_objs=(fig, axs[1]), show=False)
    plt.show()
    