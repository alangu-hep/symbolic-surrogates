#!/bin/bash

# Globals

set -xe
DATADIR=${DATADIR}/JetClass
WORKDIR=${WORKDIR}
CMD=${WORKDIR}/src/main.py
SIGNAL=TTBar
CONFIG=${WORKDIR}/data_config/JetClass/JetClass_kin.yaml
DATE=$(date +%Y%m%d_%H%M%S)

# Run-Specific

COMP=SDR

MODEL=TCVAE
TEACHER_NAME=ParT
NETWORK=${WORKDIR}/wrappers/part_wrapper.py
MODEL_PATH=${WORKDIR}/outputs/models/${SIGNAL}/ParT/ParT_diagnostic_DL_epoch-4_state.pt

DR_NETWORK=${WORKDIR}/wrappers/supervised_vae.py

MODEL_OUTPUTS=${WORKDIR}/outputs/models
METRIC_OUTPUTS=${WORKDIR}/outputs/metrics
SR_OUTPUTS=${WORKDIR}/outputs/sr_runs
LOGS=${WORKDIR}/outputs/logs

DATA_FRACTION=0.1
ALPHA=1
BETA=4
GAMMA=1
WT=0.5

suffix=${COMMENT:-default}

$CMD \
    --comp ${COMP} \
    --data-train \
    "${SIGNAL}:${DATADIR}/Pythia/train_100M/${SIGNAL}_*.root" \
    "ZJetsToNuNu:${DATADIR}/Pythia/train_100M/ZJetsToNuNu_*.root" \
    --data-val \
    "${SIGNAL}:${DATADIR}/Pythia/val_5M/${SIGNAL}_*.root" \
    "ZJetsToNuNu:${DATADIR}/Pythia/val_5M/ZJetsToNuNu_*.root" \
    --data-test \
    "${SIGNAL}:${DATADIR}/Pythia/test_20M/${SIGNAL}_*.root" \
    "ZJetsToNuNu:${DATADIR}/Pythia/test_20M/ZJetsToNuNu_*.root" \
    --data-config ${CONFIG} \
    --file-fraction 1 \
    --data-fraction ${DATA_FRACTION} \
    --batch-size 128 \
    --num-epochs 1 \
    --optimizer ranger \
    --start-lr 1e-3 \
    --lr-scheduler flat+decay \
    --model-network ${NETWORK} \
    --model-prefix ${MODEL_OUTPUTS}/${SIGNAL}/${MODEL}/${MODEL}_${COMP}_${TEACHER_NAME}_${suffix} \
    --log ${LOGS}/${SIGNAL}/${MODEL}_${COMP}_${suffix}_${DATE}.log \
    --metrics-prefix ${METRIC_OUTPUTS}/${SIGNAL}/${MODEL}_${suffix}_${DATA_FRACTION}.root \
    --kl-anneal \
    --dr-method encoder \
    --dr-network ${DR_NETWORK} \
    --alpha ${ALPHA} \
    --beta ${BETA} \
    --gamma ${GAMMA} \
    --svae-weight ${WT} 