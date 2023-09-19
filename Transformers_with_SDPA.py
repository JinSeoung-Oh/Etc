## SDPA = the scaled dot product attention

import torch
import torch.nn as nn
import torch.nn.functional as F
device = "cuda" if torch.cuda.is_available() else "cpu"

##basic
query, key, value = torch.randn(2,3,8, device=device), torch.randn(2,3,8, device=device), torch.randn(2,3,8,device=device)
F.scaled_dot_product_attention(query, key, value)

## Explicit Dispatcher Control
## If a user wants to ensure the function is indeed using the fastest implementation for their specific inputs, 
## the context manager can be used to sweep through measuring performance
## compare speed between defult, math, flash attention and memory efficient implementation 

import torch.utils.benchmark as benchmark
def benchmark_torch_function_in_microseconds(f, *args, **kwargs):
  t0 = benchmark.Timer(
        stmt="f(*args, **kwargs)", globals={"args": args, "kwargs": kwargs, "f": f}
    )
  
  return t0.blocked_autorange().mean*1e6

batch_size = 32
max_sequence_len = 1024
num_heads=32
embed_dimension=32

dtype = torch.float16

query = torch.rand(batch_size, num_heads, max_sequence_len, embed_dimension, device=device, dtype=dtype)
key = torch.rand(batch_size, num_heads, max_sequence_len, embed_dimension, device=device, dtype=dtype)
vale = torch.rand(batch_size, num_heads, max_sequence_len, embed_dimension, device=device, dtype=dtype)

print(f"The default implementation runs in {benchmark_torch_function_in_microseconds(F.scaled_dot_product_attention, query, key, value):.3f} microseconds")

# Lets explore the speed of each of the 3 implementations
from torch.backends.cuda import sdp_kernel, SDPBackend

# Helpful arguments mapper
backend_map = {
    SDPBackend.MATH: {"enable_math": True, "enable_flash": False, "enable_mem_efficient": False},
    SDPBackend.FLASH_ATTENTION: {"enable_math": False, "enable_flash": True, "enable_mem_efficient": False},
    SDPBackend.EFFICIENT_ATTENTION: {
        "enable_math": False, "enable_flash": False, "enable_mem_efficient": True}
}

with sdp_kernel(**backend_map[SDPBackend.MATH]):
    print(f"The math implementation runs in {benchmark_torch_function_in_microseconds(F.scaled_dot_product_attention, query, key, value):.3f} microseconds")


with sdp_kernel(**backend_map[SDPBackend.FLASH_ATTENTION]):
    try:
        print(f"The flash attention implementation runs in {benchmark_torch_function_in_microseconds(F.scaled_dot_product_attention, query, key, value):.3f} microseconds")
    except RuntimeError:
        print("FlashAttention is not supported. See warnings for reasons.")

with sdp_kernel(**backend_map[SDPBackend.EFFICIENT_ATTENTION]):
    try:
        print(f"The memory efficient implementation runs in {benchmark_torch_function_in_microseconds(F.scaled_dot_product_attention, query, key, value):.3f} microseconds")
    except RuntimeError:
        print("EfficientAttention is not supported. See warnings for reasons.")


## Causal self attention
class CausalSelfAttention(nn.Module):
  def __init__(self, num_heads: int, embed_dimension:int, bias:bool=False, is_causal:bool=False,dropout:float=0.0):
    super().__init__()
    assert embed_dimension % num_heads==0
    self.c_attn = nn.Linear(embed_dimension, 3*embed_dimension,bias=bias)
    self.c_proj = nn.Linear(embed_dimension,embed_dimension, bias=bias)
    self.dropout=dropout
    self.resid_dropout = nn.Dropout(dropout)
    self.num_heads = num_heads
    self.embed_dimension = embed_dimension
    self.is_causal = is_causal

  def forward(self,x):
    query_projected = self.c_attn(x)
    batch_size = query_projected.size(0)
    embed_dim = query_projected.size(2)
    head_dim = embed_dim // (self.num_heads * 3)
    query, key, value = query_projected.chunk(3,-1)
    query = query.view(batch_size, -1, self.num_heads, head_dim).transpose(1,2)
    key = key.view(batch_size, -1, self.num_heads, head_dim).transpose(1,2)
    value = value.view(batch_size, -1, self.num_heads, head_dim).transpose(1,2)
    
    if self.training:
      dropout=self.droput
      is_causal = self.is_causa;
    else:
      dropout = 0.0
      is_causal = False

    y = F.scaled_dot_product_attention(query, key, value, attn_mask=None, droput_p = dropout, is_causal= is_causal)
    y = y.transpose(1,2).view(batch_size, -1, self.num_heads * head_dim)
    y = self.resid_dropout(self.c_proj(y))
    return y


num_heads = 8
heads_per_dim = 64
embed_dimension = num_heads * heads_per_dim
dtype = torch.float16
model = CausalSelfAttention(num_heads=num_heads, embed_dimension=embed_dimension, bias=False, is_causal=True, dropout=0.1).to("cuda").to(dtype).eval()
print(model)


## NestedTensor and Dense tensor 
# NestedTensors handle the case 
# where the input is a batch of variable length sequences without needing to pad each sequence to the maximum length in the batch

import random
def generate_rand_batch(
  batch_size,
  max_sequence_len,
  embed_dimension,
  pad_percentage=None,
  dtype=torch.float16,
  device="cuda",):
    if not pad_percentage:
        return (
            torch.randn(
                batch_size,
                max_sequence_len,
                embed_dimension,
                dtype=dtype,
                device=device,
            ),
            None,
        )
    # Random sequence lengths
    seq_len_list = [
        int(max_sequence_len * (1 - random.gauss(pad_percentage, 0.01)))
        for _ in range(batch_size)
    ]
    # Make random entry in the batch have max sequence length
    seq_len_list[random.randint(0, batch_size - 1)] = max_sequence_len
    return (
        torch.nested.nested_tensor(
            [
                torch.randn(seq_len, embed_dimension,
                            dtype=dtype, device=device)
                for seq_len in seq_len_list
            ]
        ),
        seq_len_list,
    )

random_nt, _ = generate_rand_batch(32,512, embed_dimension, pad_percentage=0.5, dtype=dtype, device=device)
random_dense, _ = generate_rand_batch(32, 512, embed_dimension, pad_percentage=None, dtype=dtype, device=device)

#eval
model.eval()
with sdp_kernel(**backend_map[SDPBackend.FLASH_ATTENTION]):
  try:
      print(f"Random NT runs in {benchmark_torch_function_in_microseconds(model, random_nt):.3f} microseconds")
      print(f"Random Dense runs in {benchmark_torch_function_in_microseconds(model, random_dense):.3f} microseconds")
 except RuntimeError:
     print("FlashAttention is not supported. See warnings for reasons.")

## SDPA with TORCH.COMPILE
batch_size=32
max_sequence_len = 256
x = torch.rand(batch_size, max_sequence_len, embed_dimension, device=device, dtype=dtype)
complied_model = torch.complie(model)
compiled_model(x)
print(f"The compiled module runs in  {benchmark_torch_function_in_microseconds(compiled_model, x):.3f} microseconds")

