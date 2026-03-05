#!/bin/bash

set -e

export COMMENT=M9-RESNETKD
./m9_resnetkd.sh

export COMMENT=M10-SRRESNET
./m10_srresnet.sh