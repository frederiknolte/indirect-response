#!/bin/bash

set -e
set -x

virtualenv -p python3 .

source ./bin/activate

pip install t5[gcp]

cd google-research

pip install -r wt5/requirements.txt

python -m spacy download en_core_web_sm

cd ..

pip install ./datasets/

pip install -r ./requirements.txt

tfds build circa/circa
