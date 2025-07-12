#!/bin/csh -x

setenv TORCH_CUDA_ARCH_LIST 8.9
which python

python -u bench.py