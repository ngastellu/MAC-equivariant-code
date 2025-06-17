#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
from qcnico.plt_utils import setup_tex, get_cm
import subprocess as sbp
import os


def get_nepochs(logfile):
    print('LOGFILE = ', logfile)
    cmd = ['tail', '-n', '1', logfile]
    cmd_out =  sbp.run(cmd,stdout=sbp.PIPE)
    max_epoch = int(cmd_out.stdout.decode().split()[0])
    print(f'LOGFILE =  {logfile} -- max epoch = {max_epoch}')
    return max_epoch
    
    

def parse_losses(logfile,max_epoch=1000, log_frequency=2):

    with open(logfile) as fo:
        epoch0 = int(fo.readline().split()[0])
        npts = 1 + (max_epoch - epoch0) // log_frequency # nb of logged loss entries
        epochs = np.zeros(npts,dtype=int)
        tr_loss = np.zeros(npts)
        te_loss = np.zeros(npts)

        fo.seek(0)
        k = 0
        for line in fo:
            line = line.strip()
            if len(line) == 0:
                continue # skip empty lines
            split_line = line.split()
            epochs[k] = int(split_line[0])
            tr_loss[k] = float(split_line[1].split('(')[1][:-1]) # get rid of comma at end of number
            te_loss[k] = float(split_line[4].split('(')[1][:-1]) # get rid of comma at end of number
            k+=1
    return epochs, tr_loss, te_loss


def plot_losses_separate(epochs, tr_loss, te_loss, hyperparam_name, hyperparam_val, show=True, plt_objs=None,c_tr=None,c_te=None):
    
    fig, ax = plt_objs
    ax[0].plot(epochs, tr_loss,c=c_tr, ls='-',lw=0.8, label=f'$n = {hyperparam_val}$',)
    ax[1].plot(epochs, te_loss, c=c_te, ls='-',lw=0.8, label=f'$n = {hyperparam_val}$')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_xlim([-1,np.max(epochs) + 10])
    ax.set_ylim([0,max(np.max(tr_loss),np.max(te_loss))])
    # ax.set_title(f'{hyperparam_name} = {hyperparam_val}')
    # ax.legend()
    if show:
        # plt.savefig(f'/Users/nico/Desktop/figures_worth_saving/equivariant_MAC/losses_{hyperparam_name}_{hyperparam_val}.png')
        plt.show()
    else:
        return fig, ax


if __name__ == "__main__":

    setup_tex(fontsize=30)

    layer_type = 'equiv'

    conv_layers = [1,10,20, 40, 59]
    clrs = get_cm(conv_layers,cmap_str='magma')

    lowest_tr_losses = np.zeros(len(conv_layers))
    lowest_te_losses = np.zeros(len(conv_layers))
    fig, axs = plt.subplots(1,2,sharey=True, sharex=True)

    for k, c in enumerate(conv_layers):
        # experiment_name = f'rot_90_conv_layers_{c}'
        # logfile = f"/Users/nico/Desktop/simulation_outputs/equivariant_MAC/rot_90_conv_layer_scan_logs/{experiment_name}.log"
        experiment_name = f'rot_90_conv_layers_60_nequiv_{c}'
        logfile = f"/Users/nico/Desktop/simulation_outputs/equivariant_MAC/rot_90_equiv_first_logs/{experiment_name}.log"
        filename = os.path.basename(logfile)
        # hp_name = '_'.join(filename.split('_')[:-1])
        # hp_val = '.'.join(filename.split('_')[-1].split('.')[:-1])

        nepochs = get_nepochs(logfile)
        epochs, tr_loss, te_loss = parse_losses(logfile,max_epoch=nepochs)
        imin_tr = np.argmin(tr_loss)
        imin_te = np.argmin(te_loss)
        lowest_tr_losses[k] = tr_loss[imin_tr]
        lowest_te_losses[k] = te_loss[imin_te]

        fig, ax = plot_losses_separate(epochs, tr_loss, te_loss, 'conv_layers', c, show=False, plt_objs=(fig, ax))

        # ax.plot(epochs[imin_tr], tr_loss[imin_tr], 'ro', ms=5.0)
        # ax.vlines(epochs[imin_tr], 0, tr_loss[imin_tr], color='r', ls='--', lw=0.8)
    
        # ax.plot(epochs[imin_te], te_loss[imin_te], 'r*', ms=5.0)
        # ax.vlines(epochs[imin_te], 0, te_loss[imin_te], color='r', ls='--', lw=0.8)
        # xticks = ax.get_xticks()
        # np.append(xticks, epochs[imin_tr])
        # np.append(xticks, epochs[imin_te])
        # tick_clrs = [k] * xticks.shape[0]
        # tick_clrs[-2:] = ['r', 'r']

        # ax.set_xticks(xticks)
        # ax.set_xticklabels(xticks)
        # plt.savefig(f'/Users/nico/Desktop/figures_worth_saving/equivariant_MAC/losses_{experiment_name}.png')
 
    plt.legend()
    plt.show()
    
    plt.plot(conv_layers,lowest_tr_losses,ls='-', label='lowest training loss')
    plt.plot(conv_layers,lowest_te_losses,ls='-', label='lowest test loss')
    plt.xlabel('$N_l$')
    plt.ylabel('Lowest loss')
    plt.legend()
    plt.show()