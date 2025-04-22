#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
import os
from utils import parse_losses



def plot_losses(epochs, tr_loss, te_loss, hyperparam_name, hyperparam_val, show=True, plt_objs=None,c_tr=None,c_te=None):
    if plt_objs is None:
        fig, ax = plt.subplots()
    else:
        fig, ax = plt_objs
    ax.plot(epochs, tr_loss,c=c_tr, ls='-',lw=0.8, label='tr_loss',)
    ax.plot(epochs, te_loss, c=c_te, ls='-',lw=0.8, label='te_loss')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_xlim([-1,np.max(epochs) + 10])
    ax.set_ylim([0,max(np.max(tr_loss),np.max(te_loss))])
    ax.set_title(f'{hyperparam_name} = {hyperparam_val}')
    ax.legend()
    if show:
        plt.savefig(f'/Users/nico/Desktop/figures_worth_saving/equivariant_MAC/losses_{hyperparam_name}_{hyperparam_val}.png')
        plt.show()
    else:
        return fig, ax


if __name__ == "__main__":
    conv_layers = range(60,110,10)
    lowest_tr_losses = np.zeros(len(conv_layers))
    lowest_te_losses = np.zeros(len(conv_layers))
    for k, c in enumerate(conv_layers):
        logfile = f"/Users/nico/Desktop/simulation_outputs/equivariant_MAC/conv_layers_scan/conv_layers_{c}.log"
        filename = os.path.basename(logfile)
        hp_name = '_'.join(filename.split('_')[:-1])
        hp_val = '.'.join(filename.split('_')[-1].split('.')[:-1])

        epochs, tr_loss, te_loss = parse_losses(logfile)
        imin_tr = np.argmin(tr_loss)
        imin_te = np.argmin(te_loss)
        lowest_tr_losses[k] = tr_loss[imin_tr]
        lowest_te_losses[k] = te_loss[imin_te]

        fig, ax = plot_losses(epochs, tr_loss, te_loss, hp_name, hp_val, show=False)

        ax.plot(epochs[imin_tr], tr_loss[imin_tr], 'ro', ms=5.0)
        ax.vlines(epochs[imin_tr], 0, tr_loss[imin_tr], color='r', ls='--', lw=0.8)
    
        ax.plot(epochs[imin_te], te_loss[imin_te], 'r*', ms=5.0)
        ax.vlines(epochs[imin_te], 0, te_loss[imin_te], color='r', ls='--', lw=0.8)
        # xticks = ax.get_xticks()
        # np.append(xticks, epochs[imin_tr])
        # np.append(xticks, epochs[imin_te])
        # tick_clrs = [k] * xticks.shape[0]
        # tick_clrs[-2:] = ['r', 'r']

        # ax.set_xticks(xticks)
        # ax.set_xticklabels(xticks)
        plt.savefig(f'/Users/nico/Desktop/figures_worth_saving/equivariant_MAC/losses_{hp_name}_{hp_val}.png')
 
        plt.show()
    
    plt.plot(conv_layers,lowest_tr_losses,ls='-', label='lowest tr_loss')
    plt.plot(conv_layers,lowest_te_losses,ls='-', label='lowest te_loss')
    plt.xlabel('conv_layers')
    plt.ylabel('Lowest loss')
    plt.legend()
    plt.show()