#!/usr/bin/env bash

mkdir -p dist

if [ ! -d "packages" ]; then
    pip download -r requirements.txt -d packages
fi
if [ ! -f "dist/packages.tar.gz" ]; then
    echo "Creating packages.tar.gz"
    tar -zcf dist/packages.tar.gz packages
fi

cp requirements.txt dist/requirements.txt
cp rbcd.py dist/rbcd.py
cp -rf configs dist/configs
cp -rf inference_models dist/inference_models
