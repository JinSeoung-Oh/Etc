## Distributed Checkpoint (DCP)
# Checkpointing AI models during distributed training could be challenging, 
# as parameters and gradients are partitioned across trainers and the number of trainers available could change when you resume training.

# torch.distributed.checkpoint() enables saving and loading models from multiple ranks in parallel
# In addition, checkpointing automatically handles fully-qualified-name (FQN) mappings across models and optimizers, 
# enabling load-time resharding across differing cluster topologies

# DCP is different from torch.save() and torch.load() in a few significant ways:
# - It produces multiple files per checkpoint, with at least one per rank.
# - It operates in place, meaning that the model should allocate its data first and DCP uses that storage instead

## How to use DCP
# Saving

import os
import torch
import torch.distributed as dist
import torch.distributed.checkpoint as DCP
import torch.multiprocessing as mp
import torch.nn as nn

from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.fully_sharded_data_parallel import StateDictType

CHECKPOINT_DIR = "  "

class ToyModel(nn.Module):
    def __init__(self):
        super(ToyModel, self).__init__()
        self.net1 = nn.Linear(16, 16)
        self.relu = nn.ReLU()
        self.net2 = nn.Linear(16, 8)

    def forward(self, x):
        return self.net2(self.relu(self.net1(x)))

def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355 "

    # initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup():
    dist.destroy_process_group()

def run_fsdp_checkpoint_save_example(rank, world_size):
    print(f"Running basic FSDP checkpoint saving example on rank {rank}.")
    setup(rank, world_size)

    # create a model and move it to GPU with id rank
    model = ToyModel().to(rank)
    model = FSDP(model)

    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

    optimizer.zero_grad()
    model(torch.rand(8, 16, device="cuda")).sum().backward()
    optimizer.step()

    # set FSDP StateDictType to SHARDED_STATE_DICT so we can use DCP to checkpoint sharded model state dict
    # note that we do not support FSDP StateDictType.LOCAL_STATE_DICT
    FSDP.set_state_dict_type(
        model,
        StateDictType.SHARDED_STATE_DICT,
    )
    state_dict = {
        "model": model.state_dict(),
    }

    DCP.save_state_dict(
        state_dict=state_dict,
        storage_writer=DCP.FileSystemWriter(CHECKPOINT_DIR),
    )

    cleanup()


if __name__ == "__main__":
    world_size = torch.cuda.device_count()
    print(f"Running fsdp checkpoint example on {world_size} devices.")
    mp.spawn(
        run_fsdp_checkpoint_save_example,
        args=(world_size,),
        nprocs=world_size,
        join=True,
    )

## Loading
import os

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as DCP
import torch.multiprocessing as mp
import torch.nn as nn

from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.fully_sharded_data_parallel import StateDictType

CHECKPOINT_DIR = "checkpoint"


class ToyModel(nn.Module):
    def __init__(self):
        super(ToyModel, self).__init__()
        self.net1 = nn.Linear(16, 16)
        self.relu = nn.ReLU()
        self.net2 = nn.Linear(16, 8)

    def forward(self, x):
        return self.net2(self.relu(self.net1(x)))


def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355 "

    # initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def cleanup():
    dist.destroy_process_group()


def run_fsdp_checkpoint_load_example(rank, world_size):
    print(f"Running basic FSDP checkpoint loading example on rank {rank}.")
    setup(rank, world_size)

    # create a model and move it to GPU with id rank
    model = ToyModel().to(rank)
    model = FSDP(model)

    FSDP.set_state_dict_type(
        model,
        StateDictType.SHARDED_STATE_DICT,
    )
    # different from ``torch.load()``, DCP requires model state_dict prior to loading to get
    # the allocated storage and sharding information.
    state_dict = {
        "model": model.state_dict(),
    }

    DCP.load_state_dict(
        state_dict=state_dict,
        storage_reader=DCP.FileSystemReader(CHECKPOINT_DIR),
    )
    model.load_state_dict(state_dict["model"])

    cleanup()


if __name__ == "__main__":
    world_size = torch.cuda.device_count()
    print(f"Running fsdp checkpoint example on {world_size} devices.")
    mp.spawn(
        run_fsdp_checkpoint_load_example,
        args=(world_size,),
        nprocs=world_size,
        join=True,
    )

# ** By default, DCP saves and loads a distributed state_dict in Single Program Multiple Data(SPMD) style. 
#    To load without a distributed setup, please set no_dist to True when loading with DCP **
