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
KITTI_ROOT=/root/kitti
#CAPTUM_ROOT=/root/captum
LOGDIR=/root/logdir

# Workspace structure on host machine
# Change to path to your own machine, don't leave these lines as they are
HOST_PCDET_ROOT=/media/sg/02f4ed99-ea7d-47a9-9aed-3606f0fd7fda/tadenoud/Documents/WISEOpenLidarPerceptron
HOST_NUSC_ROOT=/media/sg/02f4ed99-ea7d-47a9-9aed-3606f0fd7fda/tadenoud/Documents/WISEOpenLidarPerceptron/pcdet/datasets/nuscenes
HOST_CADC_ROOT=/media/sg/02f4ed99-ea7d-47a9-9aed-3606f0fd7fda/tadenoud/Documents/WISEOpenLidarPerceptron/data/cadc
HOST_KITTI_ROOT=/media/sg/02f4ed99-ea7d-47a9-9aed-3606f0fd7fda/tadenoud/Documents/WISEOpenLidarPerceptron/data/kitti
#HOST_CAPTUM_ROOT=/media/sg/02f4ed99-ea7d-47a9-9aed-3606f0fd7fda/tadenoud/Documents/captum
HOST_LOGDIR=/media/sg/02f4ed99-ea7d-47a9-9aed-3606f0fd7fda/tadenoud/Documents/WISEOpenLidarPerceptron/output