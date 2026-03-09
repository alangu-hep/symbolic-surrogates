#!/bin/bash

set -e

export COMMENT=M12-PARTICLENETKD
./m12_particlenetkd.sh

export COMMENT=M13-SRPARTICLENET
./m13_srparticlenet.sh