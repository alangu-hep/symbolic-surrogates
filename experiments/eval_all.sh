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

SURROGATE_NAME=Surrogate
OBS_NAME=Observables
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
            --batch-size 64 \
            --data-fraction ${DATA_FRACTION} \
            --model-network ${NETWORK} \
            --log ${LOGS}/${SIGNAL}/${MODEL_NAME}_${suffix}_${DATE}.log \
            --dr-network ${DR_NETWORK} \
            --dr-path ${DR_PATH} \
            --surrogate-prefix ${SR_OUTPUTS}/${SIGNAL}/${SURROGATE_NAME}/${SURROGATE_NAME}_surrogates_${MODEL_NAME}-S_${DR_NAME} \
            --observable-prefix ${SR_OUTPUTS}/${SIGNAL}/${OBS_NAME}/${OBS_NAME}_observables_${DR_NAME} \
            --run-prefix ${METRIC_OUTPUTS}/${SIGNAL}/${MODEL_NAME}_${suffix}_${DR_NAME}/${MODEL_NAME} \
            --model-path ${MODEL_PATH} \
            --dl-name ${MODEL_NAME} \
            --vae-name ${DR_NAME} \
            --surrogate-name ${MODEL_NAME}-S \
            --observable-name ${OBS_NAME} \
            --surrogate-fraction 0.01 \
            --observable-fraction 0.01
    done

    EXIT_CODE=$?
    set -e

    if [ $EXIT_CODE -ne 0 ]; then
        echo "Failed: ${SIGNAL}"
    else
        echo "Completed: ${SIGNAL}"
    fi

done
