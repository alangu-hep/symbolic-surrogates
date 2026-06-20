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

MODEL=Observables
DR_NAME=BVAE
DR_NETWORK=${WORKDIR}/wrappers/vae.py

DATA_FRACTION=0.1

suffix=${COMMENT:-observables}

for i in "${!SIGNALS[@]}"; do

    SIGNAL=${SIGNALS[$i]}
    CONFIG=${WORKDIR}/data_config/JetClass/JetClass_${SIGNAL}.yaml
    DR_PATH=${WORKDIR}/outputs/models/${SIGNAL}/${DR_NAME}/${DR_NAME}_M6-BVAE_DR_epoch-4_state.pt

    echo "Signal: ${SIGNAL}"

    set +

    $CMD \
        --data-train \
        "${SIGNAL}:${DATADIR}/Pythia/train_100M/${SIGNAL}_*.root" \
        "ZJetsToNuNu:${DATADIR}/Pythia/train_100M/ZJetsToNuNu_*.root" \
        --data-val \
        "${SIGNAL}:${DATADIR}/Pythia/val_5M/${SIGNAL}_*.root" \
        "ZJetsToNuNu:${DATADIR}/Pythia/val_5M/ZJetsToNuNu_*.root" \
        --data-test \
        "${SIGNAL}:${DATADIR}/Pythia/test_20M/${SIGNAL}_*.root" \
        "ZJetsToNuNu:${DATADIR}/Pythia/test_20M/ZJetsToNuNu_*.root" \
        --batch-size 64 \
        --data-config ${CONFIG} \
        --file-fraction 1 \
        --data-fraction ${DATA_FRACTION} \
        --log ${LOGS}/${SIGNAL}/${MODEL}_${suffix}_${DATE}.log \
        --max-size 30 \
        --n-iterations 4560 \
        --n-populations 48 \
        --population-size 27 \
        --iteration-cycles 1520 \
        --dr-network ${DR_NETWORK} \
        --dr-path ${DR_PATH} \
        --observable-prefix ${SR_OUTPUTS}/${SIGNAL}/${MODEL}/${MODEL}_${suffix}_${DR_NAME} \
        --vae-name ${DR_NAME} \
        --observable-name ${MODEL} \
        --observable-fraction 0.01 

    EXIT_CODE=$?
    set -e

    if [ $EXIT_CODE -ne 0 ]; then
        echo "Failed: ${SIGNAL}"
    else
        echo "Completed: ${SIGNAL}"
    fi

done
