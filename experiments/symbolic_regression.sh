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

COMP=SR

MODEL=Surrogate
TEACHER_NAME=ParT
DR_NAME=TCVAE
NETWORK=${WORKDIR}/wrappers/part_wrapper.py
MODEL_PATH=${WORKDIR}/outputs/models/${SIGNAL}/ParT/ParT_diagnostic_DL_epoch-4_state.pt

DR_NETWORK=${WORKDIR}/wrappers/supervised_vae.py
DR_PATH=${WORKDIR}/outputs/models/${SIGNAL}/${DR_NAME}/TCVAE_SDR_ParT_diagnostic_epoch-0_state.pt

MODEL_OUTPUTS=${WORKDIR}/outputs/models
METRIC_OUTPUTS=${WORKDIR}/outputs/metrics
SR_OUTPUTS=${WORKDIR}/outputs/sr_runs
LOGS=${WORKDIR}/outputs/logs

DATA_FRACTION=0.01
ALPHA=1
BETA=4
GAMMA=1
WT=0

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
    --batch-size 64 \
    --num-epochs 1 \
    --optimizer ranger \
    --start-lr 1e-3 \
    --lr-scheduler flat+decay \
    --model-network ${NETWORK} \
    --sr-prefix ${SR_OUTPUTS}/${SIGNAL}/${MODEL}/${MODEL}_${COMP}_${TEACHER_NAME}_${DR_NAME}_${suffix} \
    --log ${LOGS}/${SIGNAL}/${MODEL}_${COMP}_${suffix}_${DATE}.log \
    --metrics-prefix ${METRIC_OUTPUTS}/${SIGNAL}/${MODEL}_${suffix}_${DATA_FRACTION}.root \
    --dr-network ${DR_NETWORK} \
    --dr-path ${DR_PATH}
    --model-path ${MODEL_PATH} \
    --max-size 40 \
    --n-iterations 20 \
    --n-populations 48 \
    --population-size 27 \
    --iteration-cycles 10 \
    --sr-loss HuberLoss(1.35)