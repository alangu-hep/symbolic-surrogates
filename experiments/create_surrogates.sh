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

SIGNALS=(
    TTBar
    WToQQ
    HToGG
)

MODELS=(
    ParT
    ResNet
    ParticleNet
)

WRAPPERS=(
    part_wrapper.py
    resnet.py
    ParticleNet.py
)

# Run-Specific

MODEL=Surrogate
DR_NAME=BVAE
DR_NETWORK=${WORKDIR}/wrappers/vae.py

DATA_FRACTION=0.1

suffix=${COMMENT:-default}

for i in "${!SIGNALS[@]}"; do

    SIGNAL=${SIGNALS[$i]}
    CONFIG=${WORKDIR}/data_config/JetClass/JetClass_${SIGNAL}.yaml
    DR_PATH=${WORKDIR}/outputs/models/${SIGNAL}/${DR_NAME}/${DR_NAME}_trained.pt

    echo "Signal: ${SIGNAL}"

    set +e

    for mod in "${!MODELS[@]}"; do
    
        MODEL_NAME=${MODELS[$mod]}
        NETWORK=${WORKDIR}/wrappers/${WRAPPERS[$mod]}
        MODEL_PATH=${WORKDIR}/outputs/models/${SIGNAL}/${MODEL_NAME}/${MODEL_NAME}_trained.pt
    
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
            --n-iterations 4560 \
            --n-populations 48 \
            --population-size 27 \
            --iteration-cycles 1520 \
            --dr-network ${DR_NETWORK} \
            --dr-path ${DR_PATH} \
            --surrogate-prefix ${SR_OUTPUTS}/${SIGNAL}/${MODEL}/${MODEL}_${suffix}_${MODEL_NAME}-S_${DR_NAME} \
            --model-path ${MODEL_PATH} \
            --dl-name ${MODEL_NAME} \
            --vae-name ${DR_NAME} \
            --surrogate-name ${MODEL_NAME}-S \
            --surrogate-fraction 0.01 
    done

    EXIT_CODE=$?
    set -e

    if [ $EXIT_CODE -ne 0 ]; then
        echo "Failed: ${SIGNAL}"
    else
        echo "Completed: ${SIGNAL}"
    fi

done
