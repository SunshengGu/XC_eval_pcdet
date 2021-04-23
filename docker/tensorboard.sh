#! /usr/bin/env bash

LOGDIR=$(cd $1 && pwd)

docker run \
    --name="tensorboard.$(whoami).$RANDOM" \
    -d \
    -p 0.0.0.0:6006:6006 \
    -v ${LOGDIR}:/root/logdir \
    --rm \
    tensorflow/tensorflow \
    tensorboard --logdir /root/logdir --bind_all