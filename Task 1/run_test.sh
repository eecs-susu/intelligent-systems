#!/bin/bash
rm -rf images
python image-generator.py 1
python detector.py
