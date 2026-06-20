#!/bin/bash

set -e

export COMMENT=surrogates
./create_surrogates.sh

export COMMENT=observables
./create_observables.sh

export COMMENT=eval
./eval_all.sh