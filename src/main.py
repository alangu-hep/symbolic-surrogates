#!/usr/bin/env python3

import sys
import os

from framework import DistillationFramework
from weaver.utils.logger import _logger, warn_n_times, _configLogger
from handlers.args import setup_argparse

from ml_utils import losses

import time

def main():

    args = setup_argparse().parse_args()

    stdout = sys.stdout
    _configLogger('weaver', stdout=stdout, filename=args.log)
    _logger.info('Started!')
    _logger.info('args:\n - %s', '\n - '.join(str(it) for it in args.__dict__.items()))

    start = time.time()

    framework = DistillationFramework(args)

    dl_loss = None
    dl_schema = {
        0: ('logits', 'preds')
    }

    framework.prepare_dl(
        split='train',
        dl_loss = dl_loss,
        dl_schema = dl_schema
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
        vae_schema = vae_schema
    )

    framework.create_eq_dataset()
    framework.fit_surrogate()

if __name__ == '__main__':
    main()