#!/usr/bin/env python3

import sys
import os

from framework import DistillationFramework
from weaver.utils.logger import _logger, warn_n_times, _configLogger
from handlers.args import setup_argparse

from ml_utils import losses

import time

def train_surrogate(args, framework):
    '''
    Creates normal surrogate
    Expects DL & VAE to exist already
    '''

    dl_loss = None
    dl_schema = {
        0: ('logits', 'preds')
    }

    framework.prepare_dl(
        split='train',
        dl_loss = dl_loss,
        dl_schema = dl_schema,
        loader = 'sur'
    )
    
    vae_loss = None
    vae_schema = {
        0: ('features', 'recon'),
        1: ('mu', 'preds'),
        2: ('log_var', 'preds')
    }

    framework.prepare_vae(
        split='train',
        vae_loss = vae_loss,
        vae_schema = vae_schema,
        loader = 'sur'
    )

    framework.create_eq_dataset('both')
    framework.fit_surrogate()

def train_observables(args, framework):
    '''
    Fits observables to VAE latents
    '''
    
    vae_loss = None
    vae_schema = {
        0: ('features', 'recon'),
        1: ('mu', 'preds'),
        2: ('log_var', 'preds')
    }

    framework.prepare_vae(
        split='train',
        vae_loss = vae_loss,
        vae_schema = vae_schema,
        loader = 'obs'
    )

    framework.create_eq_dataset('VAE')
    framework.fit_observables()

def produce_results(args, framework):
    '''
    Creates the ROOT files for an entire run
    '''

    dl_loss = None
    dl_schema = {
        0: ('logits', 'preds')
    }

    framework.prepare_dl(
        split='test',
        dl_loss = dl_loss,
        dl_schema = dl_schema,
        loader = 'gen'
    )
    
    vae_loss = None
    vae_schema = {
        0: ('features', 'recon'),
        1: ('mu', 'preds'),
        2: ('log_var', 'preds')
    }

    framework.prepare_vae(
        split='test',
        vae_loss = vae_loss,
        vae_schema = vae_schema,
        loader = 'gen'
    )

    framework.create_eq_dataset('both')

    surrogate_loss = None
    surrogate_schema = {
        0: ('logit_diff', 'preds')
    }

    framework.eval_surrogate(
        surrogate_loss=surrogate_loss,
        surrogate_schema=surrogate_schema
    )

    obs_loss = None
    obs_schema = {
        0: ('mu_vals', 'preds')
    }

    framework.eval_observables(
        obs_loss = obs_loss,
        obs_schema = obs_schema
    )

    framework.save_outputs(
        save = ['DL', 'VAE', 'SUR', 'OBS'],
        keep_inputs = ['VAE']
    )

def main():

    args = setup_argparse().parse_args()

    stdout = sys.stdout
    _configLogger('weaver', stdout=stdout, filename=args.log)
    _logger.info('Started!')
    _logger.info('args:\n - %s', '\n - '.join(str(it) for it in args.__dict__.items()))

    start = time.time()

    framework = DistillationFramework(args)

    produce_results(args, framework)

if __name__ == '__main__':
    main()