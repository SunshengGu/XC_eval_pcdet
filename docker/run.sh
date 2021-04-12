#! /usr/bin/env bash
source config.sh

docker run \
    --name="pcdet-standalone.$(whoami).$RANDOM" \
    --gpus=all \
    --rm \
    -it \
    -v "${HOST_PCDET_ROOT}":"${PCDET_ROOT}" \
    -v "${HOST_NUSC_ROOT}":"${NUSC_ROOT}" \
    -v "${HOST_CADC_ROOT}":"${CADC_ROOT}" \
    -v "${HOST_KITTI_ROOT}":"${KITTI_ROOT}" \
#    -v "${HOST_CAPTUM_ROOT}":"${CAPTUM_ROOT}" \
    -v "${HOST_LOGDIR}":"${LOGDIR}" \
    -p 5000:8888 \
    -p 5001:6006 \
    -p 5002:8889 \
    -p 5003:8890 \
    $@ \
    pcdet-standalone
