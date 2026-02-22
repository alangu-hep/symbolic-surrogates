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

COMP=KD

MODEL=ParT
NETWORK=${WORKDIR}/wrappers/part_wrapper.py
TEACHER_NETWORK=${WORKDIR}/wrappers/part_wrapper.py
TEACHER_PATH=${WORKDIR}/outputs/models/${SIGNAL}/ParT/ParT_diagnostic_DL_epoch-4_state.pt
TEACHER_NAME=ParT

MODEL_OUTPUTS=${WORKDIR}/outputs/models
METRIC_OUTPUTS=${WORKDIR}/outputs/metrics
SR_OUTPUTS=${WORKDIR}/outputs/sr_runs
LOGS=${WORKDIR}/outputs/logs

DATA_FRACTION=0.001

TEMPERATURE=3

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
    --final-lr 1e-6 \
    --lr-scheduler cos \
    --model-network ${NETWORK} \
    --model-prefix ${MODEL_OUTPUTS}/${SIGNAL}/${MODEL}/${MODEL}_${COMP}_${suffix}_${TEACHER_NAME}_${TEMPERATURE} \
    --log ${LOGS}/${SIGNAL}/${MODEL}_${COMP}_${suffix}_${DATE}.log \
    --teacher-network ${TEACHER_NETWORK} \
    --teacher-path ${TEACHER_PATH} \
    --teacher-prefix ${TEACHER_NAME} \
    --kd-temp ${TEMPERATURE}