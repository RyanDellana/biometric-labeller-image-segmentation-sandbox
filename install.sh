#!/bin/bash

# Compile eyeLike and move the shared object file.
cd ./BLISS/eyeLike
make
mv eyeLike.so ../eyeLike.so

cd ../..

# Install BLISS. (Not really necessary at this point, but kept for standardization).
python setup.py develop
