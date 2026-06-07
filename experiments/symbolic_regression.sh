#!/bin/bash

# Globals

set -xe
DATADIR=${DATADIR}/JetClass
WORKDIR=${WORKDIR}
CMD=${WORKDIR}/src/main.py
DATE=$(date +%Y%m%d_%H%M%S)
MODEL_OUTPUTS=${WORKDIR}/outputs/models
METRIC_OUTPUTS=${WORKDIR}/outputs/metrics
SR_OUTPUTS=${WORKDIR}/outputs/sr_runs
LOGS=${WORKDIR}/outputs/logs

SIGNALS=(
    TTBar
    WToQQ
    HToGG
)

# Run-Specific

COMP=SR
SIGNAL=TTBar

MODEL=Surrogate

TEACHER_NAME=ParT
NETWORK=${WORKDIR}/wrappers/part_wrapper.py

DR_NAME=BVAE
DR_NETWORK=${WORKDIR}/wrappers/vae.py

CONFIG=${WORKDIR}/data_config/JetClass/JetClass_${SIGNAL}.yaml
MODEL_PATH=${WORKDIR}/outputs/models/${SIGNAL}/${TEACHER_NAME}/${TEACHER_NAME}_M4-PARTKD_KD_PELICAN_3_epoch-4_state.pt
DR_PATH=${WORKDIR}/outputs/models/${SIGNAL}/${DR_NAME}/${DR_NAME}_M6-BVAE_DR_epoch-4_state.pt

DATA_FRACTION=0.01
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
        --num-epochs 5 \
        --optimizer ranger \
        --start-lr 1e-3 \
        --final-lr 1e-3 \
        --lr-scheduler flat+decay \
        --model-network ${NETWORK} \
        --model-prefix ${MODEL_OUTPUTS}/${SIGNAL}/${MODEL}/${MODEL}_${suffix}_${COMP} \
        --surrogate-prefix ${SR_OUTPUTS}/${SIGNAL}/${MODEL}/${MODEL}_${suffix}_${TEACHER_NAME}_${DR_NAME}_${COMP} \
        --log ${LOGS}/${SIGNAL}/${MODEL}_${suffix}_${COMP}_${DATE}.log \
        --metrics-prefix ${METRIC_OUTPUTS}/${SIGNAL}/${MODEL}_${suffix}_${DATA_FRACTION}.root \
        --max-size 40 \
        --n-iterations 4000 \
        --n-populations 48 \
        --population-size 27 \
        --iteration-cycles 1520 \
        --dr-network ${DR_NETWORK} \
        --dr-path ${DR_PATH} \
        --model-path ${MODEL_PATH} 

date +%Y%m%d_%H%M%S