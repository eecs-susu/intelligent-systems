#!/bin/bash
python image-generator.py 10
. ./remove_bad_samples.sh
python detector.py
