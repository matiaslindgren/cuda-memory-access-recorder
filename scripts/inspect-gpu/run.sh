#!/bin/sh
make
nvidia-smi -L
./bin/analyze
