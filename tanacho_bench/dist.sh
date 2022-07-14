#!/bin/sh
if [ -d dist ]; then rm -rf dist; fi
mkdir -p dist/src
cp -r config dist/config
cp cli.py dist/src/cli.py
cp predictor.py dist/src/predictor.py
cp requirements.txt dist/requirements.txt
cp -r model dist/model

zip -r dist.zip dist
