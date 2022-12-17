#!/usr/bin/env bash

zip -r -j  packages.zip packages/*
zip -r -j  inference_models.zip inference_models/*
zip -r -j  configs.zip configs/*
zip -r -j  torch-cache.zip $HOME/.cache/torch/*
