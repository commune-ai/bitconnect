#!/usr/bin/env bash

git submodule --init --recursive

cd subtensor; git checkout master; cd ..
cd backend/bittensor; git checkout master; cd ../..
