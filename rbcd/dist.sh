#!/usr/bin/env bash
rm *.zip
zip -r -j  packages.zip packages/*
zip -r -j  models.zip models/*
zip -r -j  configs.zip configs/*
zip -r -j  torch-cache.zip $HOME/.cache/torch/*
