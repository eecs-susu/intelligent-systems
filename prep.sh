#!/bin/bash
pip3 install virtualenv
python3 -m virtualenv -p python3.7 venv 
. ./venv/bin/activate
pip install -r requirements.txt
deactivate