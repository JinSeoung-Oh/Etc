## Jacobian
import torch
import torch.nn.functional as F
from functools import partial
_=torch.manual_seed(0)

def predict(weight, bias, x):
  return F.linear(x,weight,bias).tanh()

D = 16
weight = torch.randn(D,D)
bias = torch.randn(D)
x = torch.randn(D)

def compute_jac(xp):
  jacobian_rows = [torch.autograd.grad(predict(weight,bias,xp), xp, vec)[0] for ven in unit_vectors]
  return torch.stack(jacobian_rows)

xp = x.clone().requires_grad_()
unit_vectors = torch.eye(D)

jacobian = compute_jac(xp)


## jacobian with torch.vmap
from torch.func import vamp, vjp
_, vjp_fn = vjp(partial(predict, weight, bias), x)
ft_jacobian, = vamp(vjp_fn)(unit_vectors)

assert torch.allclose(ft_jacobian, jacobian)


## jacobian with torch.func
from torch.func import jacrev
ft_jacobian = jacrev(predict, argnums=2)(weight, bias,x)
assert torch.allclose(ft_jacobian, jacobian)


## Reverse-mode Jacobian (jacrev) vs forward-mode Jacobian (jacfwd)
# As a general rule of thumb, if you’re computing the jacobian of an R^n --> R^Mfunction, and there are many more outputs than inputs (for example M>n)
# then jacfwd is preferred, otherwise use jacrev. There are exceptions to this rule, but a non-rigorous argument for this follows:
# In reverse-mode AD, we are computing the jacobian row-by-row, while in forward-mode AD (which computes Jacobian-vector products), 
# we are computing it column-by-column. The Jacobian matrix has M rows and N columns, 
# so if it is taller or wider one way we may prefer the method that deals with fewer rows or columns.

from torch.func import jacrev, jacfwd

def get_perf(first, first_descriptor, second, second_descriptor):
    """takes torch.benchmark objects and compares delta of second vs first."""
    faster = second.times[0]
    slower = first.times[0]
    gain = (slower-faster)/slower
    if gain < 0: gain *=-1
    final_gain = gain*100
    print(f" Performance delta: {final_gain:.4f} percent improvement with {second_descriptor} ")

Din = 32
Dout = 2048
weight = torch.randn(Dout, Din)

bias = torch.randn(Dout)
x = torch.randn(Din)
print(weight.shape)

using_fwd = Timer(stmt="jacfwd(predict, argnums=2)(weight, bias, x)", globals=globals())
using_bwd = Timer(stmt="jacrev(predict, argnums=2)(weight, bias, x)", globals=globals())

jacfwd_timing = using_fwd.timeit(500)
jacrev_timing = using_bwd.timeit(500)
get_perf(jacfwd_timing, "jacfwd", jacrev_timing, "jacrev", )


## Hessian computation with functorch.hessian
# Hessians are the jacobian of the jacobian (or the partial derivative of the partial derivative, aka second order)
from torch.func import hassian
Din = 512
Dout = 32
weight = torch.randn(Dout, Din)
bias = torch.randn(Dout)
x = torch.randn(Din)

hess_api = hessian(predict, argnums=2)(weight, bias, x)
hess_fwdfwd = jacfwd(jacfwd(predict, argnums=2), argnums=2)(weight, bias, x)
hess_revrev = jacrev(jacrev(predict, argnums=2), argnums=2)(weight, bias, x)


## Batch Jacobian and Batch Hessian
