#!/usr/bin/env bash
#!/bin/bash
export ENV_NAME=khan_internal
export tensorflow=tensorflow-gpu

export CONDA_EXISTS=`which conda`
if [ "$CONDA_EXISTS" = "" ];
then
    wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O anaconda.sh;
    bash anaconda.sh -b -p `pwd`/anaconda
    export PATH=`pwd`/anaconda/bin:$PATH
else
    echo "Using Existing Conda"
fi

# Install Libraries
conda create -y --name $ENV_NAME python=3.5
conda config --add channels anaconda
conda config --add channels conda-forge
source activate $ENV_NAME

if [[ "$unamestr" == 'Darwin' ]]; then
   export tensorflow=tensorflow
fi

yes | pip install tensorflow-gpu==1.9.0
yes | pip install nose

echo "Installed $ENV_NAME conda environment"