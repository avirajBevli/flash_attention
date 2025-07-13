import math
import nvtx
import torch
from torch.nn import functional as F
from torch.utils.cpp_extension import load
import time

# avoid torch+GPU weird errors if torch does not find GPU!
assert torch.cuda.is_available()

import sys
# print("SYS PATH: ", sys.path, "\n")
print("Python executable being used:", sys.executable, "\n")

# Have to make sure that the .so file is compiled first! 
start_time = time.time()
import importlib.util
import sys
import os
module_path = os.path.expanduser(
    '~/.cache/torch_extensions/py312_cu126/custom_flash_attention/custom_flash_attention.so'
)
spec = importlib.util.spec_from_file_location("custom_flash_attention", module_path)
custom_flash_attention = importlib.util.module_from_spec(spec)
sys.modules["custom_flash_attention"] = custom_flash_attention
spec.loader.exec_module(custom_flash_attention)
print(" =========== Extention load time: ", time.time() - start_time)




# Use small model params, otherwise slower than manual attention. See caveats in README.
batch_size = 16
n_head = 12
seq_len = 64
head_embd = 64

q = torch.randn(batch_size, n_head, seq_len, head_embd).cuda()
k = torch.randn(batch_size, n_head, seq_len, head_embd).cuda()
v = torch.randn(batch_size, n_head, seq_len, head_embd).cuda()

print("device of q, k, v: ", q.device, k.device, v.device)

# print('=== profiling manual attention ===')
# Our minimal flash attention aims to be faster than this by avoiding HBM read/writes of N^2 matrices.
def naiive_attn(q, k, v):
    att = (q @ k.transpose(-2, -1) * (1.0 / math.sqrt(k.size(-1))))
    att = F.softmax(att, dim=-1)
    y = att @ v
    return y

# with torch.autograd.profiler.profile(use_cuda=True) as prof:
print("Running naiive attention")
with nvtx.annotate("naiive", color='yellow'):
    manual_result = naiive_attn(q, k, v)
# print(prof.key_averages().table(sort_by='cuda_time_total', row_limit=10))

# print('=== profiling minimal flash attention === ')
# with torch.autograd.profiler.profile(use_cuda=True) as prof:
with nvtx.annotate("custom", color='green'):
    minimal_result = custom_flash_attention.forward(q, k, v)
# print(prof.key_averages().table(sort_by='cuda_time_total', row_limit=10))

print(' \n \n ========== attn values sanity check:', 
    torch.allclose(minimal_result, manual_result, rtol=0, atol=1e-02),
    " ================ \n \n ")