#!/bin/bash

# download OpenFOAM and ThirdParty
wget https://dl.openfoam.com/source/v2306/OpenFOAM-v2306.tgz
wget https://dl.openfoam.com/source/v2306/ThirdParty-v2306.tgz

# unpack OpenFOAM and ThirdParty
tar zxvf OpenFOAM-v2306.tgz
tar zxvf ThirdParty-v2306.tgz

# compile
cd OpenFOAM-v2306
source etc/bashrc
./Allwmake -j -q -l
cd ..