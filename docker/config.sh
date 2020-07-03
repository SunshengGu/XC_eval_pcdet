#! /usr/bin/env bash

# Build component versions
CMAKE_VERSION=3.16
CMAKE_BUILD=5
PYTHON_VERSION=3.6
TORCH_VERSION=1.2
TORCHVISION_VERSION=0.4

# Workspace structure in docker
PCDET_ROOT=/root/pcdet
NUSC_ROOT=/root/nusc
CADC_ROOT=/root/cadc
LOGDIR=/root/logdir

# Workspace structure on host machine
HOST_PCDET_ROOT=/path/to/pcdet
HOST_NUSC_ROOT=/path/to/nuscenes
HOST_CADC_ROOT=/path/to/cadc
HOST_LOGDIR=/path/to/logdir
