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
)

MODELS=(
    ParT
)

EXPERIMENTS=(
    M1-CONTROL
)

WRAPPERS=(
    part_wrapper.py
)

# Run-Specific

MODEL=Surrogate
DR_NAME=BVAE
DR_NETWORK=${WORKDIR}/wrappers/vae.py

DATA_FRACTION=0.01

suffix=${COMMENT:-default}

for i in "${!SIGNALS[@]}"; do

    SIGNAL=${SIGNALS[$i]}
    CONFIG=${WORKDIR}/data_config/JetClass/JetClass_${SIGNAL}.yaml
    DR_PATH=${WORKDIR}/outputs/models/${SIGNAL}/${DR_NAME}/${DR_NAME}_M6-BVAE_DR_epoch-4_state.pt

    echo "Signal: ${SIGNAL}"

    set +e

    for mod in "${!MODELS[@]}"; do
    
        MODEL_NAME=${MODELS[$mod]}
        EXPERIMENT=${EXPERIMENTS[$mod]}
        NETWORK=${WORKDIR}/wrappers/${WRAPPERS[$mod]}
        
        MODEL_PATH=${WORKDIR}/outputs/models/${SIGNAL}/${MODEL_NAME}/${MODEL_NAME}_${EXPERIMENT}_DL_epoch-4_state.pt
    
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
            --data-config ${CONFIG} \
            --file-fraction 1 \
            --data-fraction ${DATA_FRACTION} \
            --model-network ${NETWORK} \
            --log ${LOGS}/${SIGNAL}/${MODEL}_${suffix}_${DATE}.log \
            --max-size 40 \
            --n-iterations 5000 \
            --n-populations 50 \
            --population-size 30 \
            --iteration-cycles 1520 \
            --dr-network ${DR_NETWORK} \
            --dr-path ${DR_PATH} \
            --surrogate-prefix ${SR_OUTPUTS}/${SIGNAL}/${MODEL}/${MODEL}_${suffix}_${MODEL_NAME}-S_${DR_NAME} \
            --model-path ${MODEL_PATH} \
            --dl-name ${MODEL_NAME} \
            --vae-name ${DR_NAME} \
            --surrogate-name ${MODEL_NAME}-S \
            --surrogate-fraction 0.1 
    done

    EXIT_CODE=$?
    set -e

    if [ $EXIT_CODE -ne 0 ]; then
        echo "Failed: ${SIGNAL}"
    else
        echo "Completed: ${SIGNAL}"
    fi

done