FROM nvidia/cuda:10.1-cudnn7-devel-ubuntu18.04
ENV LANG C.UTF-8
ENV LC_ALL C.UTF-8

ARG CMAKE_VERSION=3.16
ARG CMAKE_BUILD=5
ARG PYTHON_VERSION=3.6
ARG TORCH_VERSION=1.2
ARG TORCHVISION_VERSION=0.4

RUN APT_INSTALL="apt-get install -y --no-install-recommends" && \
    PIP_INSTALL="python -m pip --no-cache-dir install --upgrade" && \
    GIT_CLONE="git clone --recursive" && \

    rm -rf /var/lib/apt/lists/* \
           /etc/apt/sources.list.d/cuda.list \
           /etc/apt/sources.list.d/nvidia-ml.list && \

    apt-get update && \

# ==================================================================
# tools
# ------------------------------------------------------------------

    DEBIAN_FRONTEND=noninteractive $APT_INSTALL \
        build-essential \
        apt-utils \
        ca-certificates \
        wget \
        git \
        vim \
        libssl-dev \
        curl \
        unzip \
        unrar \
        && \

    wget -O ~/cmake.tar.gz \
        https://cmake.org/files/v${CMAKE_VERSION}/cmake-${CMAKE_VERSION}.${CMAKE_BUILD}.tar.gz && \
    tar -C ~/ -xzf ~/cmake.tar.gz && \
    cd ~/cmake-${CMAKE_VERSION}.${CMAKE_BUILD} && \
    ./bootstrap && \
    make -j$(nproc) install && \

# ==================================================================
# python
# ------------------------------------------------------------------

    DEBIAN_FRONTEND=noninteractive $APT_INSTALL \
        software-properties-common \
        && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && \
    DEBIAN_FRONTEND=noninteractive $APT_INSTALL \
        python${PYTHON_VERSION} \
        python${PYTHON_VERSION}-dev \
        python3-distutils-extra \
        && \
    wget -O ~/get-pip.py \
        https://bootstrap.pypa.io/get-pip.py && \
    python3.6 ~/get-pip.py && \
    ln -s /usr/bin/python${PYTHON_VERSION} /usr/local/bin/python3 && \
    ln -s /usr/bin/python${PYTHON_VERSION} /usr/local/bin/python && \

# ==================================================================
# dependencies
# ------------------------------------------------------------------

    $PIP_INSTALL \
        cachetools \
        easydict \
        enum34 \
        fire \
        future \
        h5py \
        motmetrics \
        numba \
        numpy \
        opencv-python \
        pillow \
        protobuf \
        psutil \
        pybind11 \
        pyquaternion \
        pyyaml \
        seaborn \
        scikit-learn \
        scikit-image \
        scipy \
        setuptools \
        shapely \
        tensorboardX \
        tqdm \
        typing \
        && \

# ==================================================================
# pytorch
# ------------------------------------------------------------------

    $PIP_INSTALL \
        torch==$TORCH_VERSION \
        torchvision==$TORCHVISION_VERSION \
        && \

# ==================================================================
# spconv
# ------------------------------------------------------------------

    DEBIAN_FRONTEND=noninteractive $APT_INSTALL \
        libsm6 libxext6 libxrender-dev \
        libboost-all-dev && \

    $GIT_CLONE https://github.com/traveller59/spconv.git \
        ~/spconv && \
    cd ~/spconv && \
    git checkout 8da6f967fb9a054d8870c3515b1b44eca2103634 && \
    SPCONV_FORCE_BUILD_CUDA=1 \
    python setup.py bdist_wheel && \
    ls -la ~/spconv/dist/ && \
    $PIP_INSTALL ~/spconv/dist/spconv-*.whl && \

# ==================================================================
# config & cleanup
# ------------------------------------------------------------------

    cd && \
    ldconfig && \
    apt-get clean && \
    apt-get autoremove && \
    rm -rf /var/lib/apt/lists/* \
        /tmp/* \
        ~/*


ARG PCDET_ROOT=/root/pcdet
ENV PCDET_ROOT=$PCDET_ROOT

ARG NUSC_ROOT=/root/nusc
ENV NUSC_ROOT=$NUSC_ROOT

ARG CADC_ROOT=/root/cadc
ENV CADC_ROOT=$CADC_ROOT

ARG LOGDIR=/root/logdir
ENV LOGDIR=$LOGDIR

ENV PYTHONPATH=$PYTHONPATH:$PCDET_ROOT
ENV PYTHONPATH=$PYTHONPATH:$NUSC_ROOT/nuscenes-devkit/python-sdk

# TODO: Create dataset symlinks for datasets

VOLUME ["$PCDET_ROOT", "$CADC_ROOT", "$NUSC_ROOT", "$LOGDIR"]
WORKDIR /root/

ENTRYPOINT ["bash"]
