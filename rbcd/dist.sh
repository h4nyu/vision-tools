#!/usr/bin/env bash
rm *.zip
if [ ! -d "packages" ]; then
    pip download -r requirements.txt -d packages
fi
zip -r -j  packages.zip packages/*
zip -r -j  models.zip models/*
zip -r -j  configs.zip configs/*
zip -r -j  torch-cache.zip $HOME/.cache/torch/*
