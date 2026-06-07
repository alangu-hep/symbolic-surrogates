#!/bin/bash

# Globals

set -x
DATADIR=${DATADIR}/JetClass
WORKDIR=${WORKDIR}
CMD=${WORKDIR}/src/main.py
DATE=$(date +%Y%m%d_%H%M%S)
MODEL_OUTPUTS=${WORKDIR}/outputs/models
METRIC_OUTPUTS=${WORKDIR}/outputs/metrics
SR_OUTPUTS=${WORKDIR}/outputs/sr_runs
LOGS=${WORKDIR}/outputs/logs
FIGURES=${WORKDIR}/figures

SIGNALS=(
    TTBar
    WToQQ
    HToGG
)

# Run-Specific

COMP=SR

MODEL=Surrogate
MODEL_NAME=ParT
NETWORK=${WORKDIR}/wrappers/part_wrapper.py
TEACHER_NAME=PELICAN
DR_NAME=BVAE
DR_NETWORK=${WORKDIR}/wrappers/vae.py

DATA_FRACTION=0.01

TEMPS=(
    3
    7
    5
)

suffix=${COMMENT:-default}

for i in "${!SIGNALS[@]}"; do

    SIGNAL=${SIGNALS[$i]}
    TEMPERATURE=${TEMPS[$i]}
    CONFIG=${WORKDIR}/data_config/JetClass/JetClass_${SIGNAL}.yaml
    MODEL_PATH=${WORKDIR}/outputs/models/${SIGNAL}/${MODEL_NAME}/${MODEL_NAME}_M4-PARTKD_KD_${TEACHER_NAME}_${TEMPERATURE}_epoch-4_state.pt
    DR_PATH=${WORKDIR}/outputs/models/${SIGNAL}/${DR_NAME}/${DR_NAME}_M6-BVAE_DR_epoch-4_state.pt
    echo "Signal: ${SIGNAL}"

    set +e
    
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
        --log ${LOGS}/${SIGNAL}/${MODEL}_${suffix}_${COMP}_${DATE}.log \
        --max-size 40 \
        --n-iterations 4560 \
        --n-populations 48 \
        --population-size 27 \
        --iteration-cycles 1520 \
        --dr-network ${DR_NETWORK} \
        --dr-path ${DR_PATH} \
        --sr-prefix ${SR_OUTPUTS}/${SIGNAL}/${MODEL}/${MODEL}_${suffix}_${MODEL_NAME}_${DR_NAME}_${COMP} \
        --model-path ${MODEL_PATH}

    EXIT_CODE=$?
    set -e

    if [ $EXIT_CODE -ne 0 ]; then
        echo "Failed: ${SIGNAL}"
    else
        echo "Completed: ${SIGNAL}"
    fi

done