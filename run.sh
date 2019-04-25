#!/bin/bash

# Launches a docker container using our image, and runs torch
gpu=$1
shift

NV_GPU=$gpu docker run --runtime=nvidia --rm -ti \
        -v `pwd`/code:/project \
        $USER/comm \
        $@
