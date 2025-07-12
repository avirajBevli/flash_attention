#!/bin/csh -x

setenv TORCH_CUDA_ARCH_LIST 8.9
echo "About to run Nsight profiling with sudo..."

sudo --preserve-env=PATH,LD_LIBRARY_PATH,CUDA_HOME \
	ncu \
	--kernel-name='forward_kernel' \
	--force-overwrite \
	-o ./profiles/v0_ncu \
	python bench.py

echo "===== DONE running Nsight profiling"
