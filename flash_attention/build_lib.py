import torch
from torch.utils.cpp_extension import load
import time

# Load the CUDA kernel as a python module
start_time = time.time()
custom_flash_attention = load(
    name='custom_flash_attention', 
    sources=['main.cpp', 'flash.cu'], 
    extra_cuda_cflags=['-O2'],
)
print(" ============== TIme to compile .so file: ", time.time() - start_time, " ============= ")
