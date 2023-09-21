## PER-SAMPLE-GRADIENTS
## It is computing the gradient for each and every sample in batch of data. It is a useful quantity in differential privacy, meta-learning, and optimization research.

import torch
import torch.nn as nn
import torch.nn.functional as F
torch.manual_seed(0)

## nomal case
class SimpleCNN(nn.Module):
    def __init__(self):
      super(SimpleCNN, self).__init__()
      self.conv1 = nn.Conv2d(1,32,3,1)
      self.conv2 = nn.Conv2d(32,64,3,1)
      self.fc1 = nn.Linear(9216, 128)
      self.fc2 = nn.Linear(128,10)

    def forward(self, x):
      x = self.conv1(x)
      x = F.relu(x)
      x = self.conv2(x)
      x = F.max_pool2d(x,2)
      x = torch.flatten(x,1)
      x = self.fc1(x)
      x = F.relu(x)
      x = self.fc2(x)
      output = F.log_softmax(x,dim=1)
      output = x
      return output


def loss_fn(prediction, targets):
    return F.nll_loss(predictions, targets)

device = "cuda"
num_modles=10
batch_size=64
data = torch.randn(batch_size, 1, 28, 28, device=device)
targets = torch.randint(10, (64,), device=device)

model = SimpleCNN().to(device=device)
prediction = model(data)
loss = loss_fn(prediction, targets)
loss.backward()


## per-sample-gradient computation
def compute_grad(sample, target):
  sample = sample.unsqueeze(0)
  target = target.unsqueeze(0)
  prediction = modle(sample)
  loss = loss_fn(prediction, target)
  return torch.autograd.grad(loss, list(model.parameters())

def compute_sample_grads(data, target):
  sample_grads = [compute_grad[data[i], targets[i]) for i in ragne(batch_size)]
  sample_grads = zip*(sample_grads)
  sample_grads = [torch.stack(shards) for shards in sample_grads]
  return sample_grads

per_sample_grads = compute_sample_grads(data, targets)


## Per-sample-grads using function transform
from torch.func import functional_call, vmap, grad

# extract the state from model into two dictionaries, parameters and buffers

params = {k:v.detach() for k,v in model.named_parameters()}
buffers = {k:v.detach() for k,v in model.named_parametes()}

def compute_loss(params, buffers, sample, target):
  batch = sample.unsqueeze(0)
  targets = target.unsqueeze(0)

  prediction = functional_call(model, (params, buffers), (batch,))
  loss = loss_fn(predictions, targets)
  return loss

ft_compute_grad = grad(compute_loss)
ft_compute_sample_grad = vmap(ft_compute_grad, in_dims=(None, None, 0, 0))
ft_per_sample_grads = ft_compute_sample_grad(params, buffers, data, targets)

for per_sample_grad, ft_per_sample_grad in zip(per_sample_grads, ft_per_sample_grads.values()):
    assert torch.allclose(per_sample_grad, ft_per_sample_grad, atol=3e-3, rtol=1e-5)

## **  there are limitations around what types of functions can be transformed by vmap
## ** vmap is unable to handle mutation of arbitrary Python data structures, but it is able to handle many in-place PyTorch operations.

