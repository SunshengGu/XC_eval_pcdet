#! /usr/bin/env bash

# Ubuntu 18.04
# Step 0: Clone repo
echo "Enter FULL path of where to clone spconv"
read spconv_path
cd $spconv_path
git clone git@github.com:traveller59/spconv.git --recursive
# checkout older spconv 1.0
cd $spconv_path/spconv/
git checkout 8da6f967fb9a054d8870c3515b1b44eca2103634
# Step 1: Install boost headers
sudo apt-get install libboost-all-dev
# Step 2:
cmake --version
echo "if the cmake version above is < 3.13.2 then run update_cmake.bash!"
echo "press enter"
read tmp
# Step 3: Run set up
pip show torch
echo "torch should be > 1.0!"
echo "press enter"
read tmp
python setup.py bdist_wheel
# Step 4: pip install the generated whl file
cd $spconv_path/spconv/dist
pip install spconv*.whl

