## torch.compile makes PyTorch code run faster by JIT-compiling PyTorch code into optimized kernels,
## all while requiring minimal code changes

## check_gpu
import torch
import warnings

gpu_ok=False
if torch.cuda.is_available():
  device_cap = torch.cuda.get_device_capability()
  if device_cap in ((7,0), (8,0), (9,0)):
      gpu_ok = True
if not gpu_ok:
  warnings.warn(
        "GPU is not NVIDIA V100, A100, or H100. Speedup numbers may be lower "
        "than expected."
    )

## basic usage

def foo(x,y):
  a = torch.sin(x)
  b = torch.cos(y)
  return a+b
opt_foo1 = torch.compile(foo)
print(opt_foo1(torch.randn(10, 10), torch.randn(10, 10)))

# or
@torch.compile
def opt_foo2(x,y):
  a = torch.sin(x)
  b = torch.cos(y)
  return a+b
print(opt_foo2(torch.randn(10, 10), torch.randn(10, 10)))

## optimize torch.nn.module instances
class MyModule(torch.nn.Module):
  def __init__(self):
    super().__init__()
    self.lin = torch.nn.Linear(100,10)
  def forward(self,x):
    return torch.nn.functional.relu(self.lin(x))

mod = MyModule()
opt_mod = torch.compile(mod)


## Demostrating Sppedups
def timed(fn):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    result = fn()
    end.record()
    torch.cuda.synchronize()
    return result, start.elapsed_time(end) / 1000

# Generates random input and targets data for the model, where `b` is batch size.
def generate_data(b):
    return (
        torch.randn(b, 3, 128, 128).to(torch.float32).cuda(),
        torch.randint(1000, (b,)).cuda(),
    )

N_ITERS = 10

from torchvision.models import densenet121
def init_model():
    return densenet121().to(torch.float32).cuda()

def evaluate(mod, inp):
  with torch.no_grad():
      return mod(inp)


model = init_model()
torch._dynamo.reset()

evaluate_opt = torch.compile(evaluate, mode="reduce-overhead")

inp = generate_data(16)[0]
print("eager:", timed(lambda: evaluate(model, inp))[1])
print("compile:", timed(lambda: evaluate_opt(model, inp))[1])

eager_times = []
for i in range(N_ITERS):
    inp = generate_data(16)[0]
    _, eager_time = timed(lambda: evaluate(model, inp))
    eager_times.append(eager_time)
    print(f"eager eval time {i}: {eager_time}")

print("~" * 10)

compile_times = []
for i in range(N_ITERS):
    inp = generate_data(16)[0]
    _, compile_time = timed(lambda: evaluate_opt(model, inp))
    compile_times.append(compile_time)
    print(f"compile eval time {i}: {compile_time}")
print("~" * 10)

import numpy as np
eager_med = np.median(eager_times)
compile_med = np.median(compile_times)
speedup = eager_med / compile_med
print(f"(eval) eager median: {eager_med}, compile median: {compile_med}, speedup: {speedup}x")
print("~" * 10)


### 
