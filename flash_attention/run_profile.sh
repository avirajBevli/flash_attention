#!/bin/bash

# run build_lib.py to build the .so file first
python build_lib.py


sudo --preserve-env=PATH,VIRTUAL_ENV,LD_LIBRARY_PATH,CUDA_HOME \
	ncu --kernel-name 'forward_kernel' \
	--force-overwrite \
	-o ./profile_reports/ncu_v0 \
	python bench.py
echo -e "===== Nsight Compute profiling done ====="


nsys profile \
	--trace=cuda,nvtx \
	--force-overwrite=true \
	-o ./profile_reports/nsys_v0 \
	python bench.py
echo -e "===== Nsight Systems profiling done ====="
