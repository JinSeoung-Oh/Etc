## https://pytorch.org/tutorials/prototype/pt2e_quant_qat.html
## how to perform quantization-aware training (QAT) in graph mode based on torch.export.export

import torch
from torch._export import capture_pre_autograd_graph
from torch.ao.quantization.quantize_pt2e import (
  prepare_qat_pt2e,
  convert_pt2e,
)
from torch.ao.quantization.quantizer import (
  XNNPACKQuantizer,
  get_symmetric_quantization_config,
)

class M(torch.nn.Module):
   def __init__(self):
      super().__init__()
      self.linear = torch.nn.Linear(5, 10)

   def forward(self, x):
      return self.linear(x)


example_inputs = (torch.randn(1, 5),)
m = M()

## Step 1. program capture
m = capture_pre_autograd_graph(m, *example_inputs)
# or to caputre with dynamic dimensions:
# from torch.export import dynamic_dim
# example_inputs = (torch.rand(2, 3, 224, 224),)
# exported_model = capture_pre_autograd_graph(float_model, example_inputs, constraints=[dynamic_dim(example_inputs[0], 0)])

## Step 2. quantization-aware training
# backend developer will write their own Quantizer and expose methods to allow
# users to express how they want the model to be quantized
# Quantizer is backend specific, and each Quantizer will provide their own way to allow users to configure their model
quantizer = XNNPACKQuantizer().set_global(get_symmetric_quantization_config(is_qat=True))

## Step 3. Prepare the Model for Quantization-Aware Training
# prepare_qat_pt2e inserts fake quantizes in appropriate places in the model and performs the appropriate QAT “fusions”, 
# such as Conv2d + BatchNorm2d, for better training accuracies
prepared_model = prepare_qat_pt2e(m, quantizer)

#* In the Beta version, if model contains batch normalization then if CPU, code have to get torch.ops.aten._native_batch_norm_legit
#* If GPU, code have to get torch.ops.aten.cudnn_batch_norm

## Step 4. training loop (example)

num_epochs = 10
num_train_batches = 20
num_eval_batches = 20
num_observer_update_epochs = 4
num_batch_norm_update_epochs = 3
num_epochs_between_evals = 2

# QAT takes time and one needs to train over a few epochs.
# Train and check accuracy after each epoch
for nepoch in range(num_epochs):
    train_one_epoch(prepared_model, criterion, optimizer, data_loader, "cuda", num_train_batches)

    # Optionally disable observer/batchnorm stats after certain number of epochs
    if epoch >= num_observer_update_epochs:
        print("Disabling observer for subseq epochs, epoch = ", epoch)
        prepared_model.apply(torch.ao.quantization.disable_observer)
    if epoch >= num_batch_norm_update_epochs:
        print("Freezing BN for subseq epochs, epoch = ", epoch)
        for n in prepared_model.graph.nodes:
            # Args: input, weight, bias, running_mean, running_var, training, momentum, eps
            # We set the `training` flag to False here to freeze BN stats
            if n.target in [
                torch.ops.aten._native_batch_norm_legit.default,
                torch.ops.aten.cudnn_batch_norm.default,
            ]:
                new_args = list(n.args)
                new_args[5] = False
                n.args = new_args
        prepared_model.recompile()

    if (nepoch + 1) % num_epochs_between_evals == 0:
        prepared_model_copy = copy.deepcopy(prepared_model)
        quantized_model = convert_pt2e(prepared_model_copy)
        top1, top5 = evaluate(quantized_model, criterion, data_loader_test, neval_batches=num_eval_batches)
        print('Epoch %d: Evaluation accuracy on %d images, %2.2f' % (nepoch, num_eval_batches * eval_batch_size, top1.avg))

## model save
checkpoint_path = "/path/to/my/checkpoint_%s.pth" % nepoch
torch.save(prepared_model.state_dict(), "checkpoint_path")


## model load
from torch._export import capture_pre_autograd_graph
from torch.ao.quantization.quantizer.xnnpack_quantizer import (
    XNNPACKQuantizer,
    get_symmetric_quantization_config,
)
from torchvision.models.resnet import resnet18

example_inputs = (torch.rand(2, 3, 224, 224),)
float_model = resnet18(pretrained=False)
exported_model = capture_pre_autograd_graph(float_model, example_inputs)
quantizer = XNNPACKQuantizer()
quantizer.set_global(get_symmetric_quantization_config(is_qat=True))
prepared_model = prepare_qat_pt2e(exported_model, quantizer)
prepared_model.load_state_dict(torch.load(checkpoint_path)
                               
## Convert the Trained Model to a Quantized Model
#convert_pt2e takes a calibrated model and produces a quantized model. Note that, before inference, you must first 
#call torch.ao.quantization.move_exported_model_to_eval() to ensure certain ops like dropout behave correctly in the eval graph
quantized_model = convert_pt2e(prepared_model)
# we have a model with aten ops doing integer computations when possible

# move the quantized model to eval mode, equivalent to `m.eval()`
torch.ao.quantization.move_exported_model_to_eval(quantized_mode)
torch.ao.quantization.move_exported_model_to_eval(m)

print(quantized_model)

top1, top5 = evaluate(quantized_model, criterion, data_loader_test, neval_batches=num_eval_batches)
print('Final evaluation accuracy on %d images, %2.2f' % (num_eval_batches * eval_batch_size, top1.avg))

#** Have to call torch.ao.quantization.move_exported_model_to_eval() instead of model.eval()
#** Have to call torch.ao.quantization.move_exported_model_to_train() instead of model.train()
#** model.eval() and model.train --> no longer correctly change the behavior of certain ops like dropout and batch normalization

