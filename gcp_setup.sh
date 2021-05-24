#!/bin/bash

set -e
set -x

virtualenv -p python3 .

source ./bin/activate

pip install --upgrade pip

pip install t5[gcp]

cd google-research

pip install -r wt5/requirements.txt

python -m spacy download en_core_web_sm

cd ..

pip install ./datasets/

pip install -r LAS/requirements.txt

cp -r circa/ google-research/wt5/circa

pip install -r requirements.txt
