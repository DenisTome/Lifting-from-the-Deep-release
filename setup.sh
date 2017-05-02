#!/bin/bash

mkdir saved_sessions
cd saved_sessions

echo 'Downloading models...'
wget http://visual.cs.ucl.ac.uk/pubs/liftingFromTheDeep/res/init_session.tar.gz
wget http://visual.cs.ucl.ac.uk/pubs/liftingFromTheDeep/res/prob_model.tar.gz

echo 'Extracting models...'
tar -xvzf init_session.tar.gz
tar -xvzf prob_model.tar.gz
rm -rf init_session.tar.gz
rm -rf prob_model.tar.gz
cd ..

echo 'Installing dependencies...'
pip install Cython
pip install scikit-image

echo 'Compiling external utilities...'
cd utils/external/
python setup_fast_rot.py build
cd ../../
ln -sf utils/external/build/lib.linux-x86_64-2.7/upright_fast.so ./

echo 'Done'
