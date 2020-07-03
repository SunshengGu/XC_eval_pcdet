#! /usr/bin/env bash

# Remove current version
sudo apt purge --auto-remove cmake
# https://askubuntu.com/questions/355565/how-do-i-install-the-latest-version-of-cmake-from-the-command-line
version=3.16
build=5
mkdir ~/temp
cd ~/temp
wget https://cmake.org/files/v$version/cmake-$version.$build.tar.gz
tar -xzvf cmake-$version.$build.tar.gz
cd cmake-$version.$build/
# install
cd ~/temp/cmake-$version.$build
./bootstrap
make -j$(nproc)
sudo make install
# Check the version
cmake --version

