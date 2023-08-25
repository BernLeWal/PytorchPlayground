# PyTorch-Playground

Repo to investigate and try out Pytorch.

Goal is to have all the data and run the models locally. So a good GPU is required. (I used a Nvidia RTX 2080TI)

Contents:
* Einstieg in PyTorch basierend auf dem iX-Artikel: ix.de/zbu8


## Pre-Requisites

### Pytorch base-system
* Python 3 (=3.10! Not 3.11!) + VENV
* Visual Studio Code + Python Extension
* Git
* Microsoft Visual C++ Redistributable for Visual Studio 2022 

### Install
* Nvidia-Drivers for your GPU
* Evaluate the CUDA version with the following command: ```nvidia-smi``` (should be above 11)
* Setup python virtual environment (venv)
```shell
py -3.10 -m pip install virtualenv
py -3.10 -m virtualenv .venv
.venv\Scripts\activate

pip3 install -r requirements.txt
```
* Get install command from the following website, using the CUDA version from the nvidia-smi output:
https://pytorch.org/get-started/locally/  
(on 2023-08-04 for my RTX2080 it was:
```pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118```  )

### Files used in Samples 
(to be placed into the /data directory):
* Dataset: language-identification  
see https://huggingface.co/datasets/papluca/language-identification/tree/main

## Running

Select one of the sampleXY.py files and execute them.

## Tools

CSV to JSON - CSVJSON online tool: https://csvjson.com/csv2json