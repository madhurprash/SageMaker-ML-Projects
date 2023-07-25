import argparse
import json
import logging
import os
import sys

import torch
## Represents importing the distributed code and used to prepare the model
## for distributed training.
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data

## This is used to wrap the model to tell pytorch that this is a distributed training
import torch.utils.data.distributed
import torchvision
from torchvision import datasets, transforms

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))

# Defining models
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Define data augmentation
def _get_transforms():
        transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
        return transform

# Define data loader for training dataset
def _get_train_data_loader(batch_size, training_dir, is_distributed):
    logger.info("Get train data loader")
    

   
    train_set = torchvision.datasets.CIFAR10(root=training_dir,
                                             ## Here, this script runs on a container in a SM cluster. SM will pass this training directory information through the scipt as a command line argument
                                             train=True, 
                                             download=False, 
                                             transform=_get_transforms()) 
    
    train_sampler = (
        ## Distributed sampler runs it in distributed fashion
        torch.utils.data.distributed.DistributedSampler(train_set) if is_distributed else None
    )
    
    return torch.utils.data.DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=train_sampler is None,
        sampler=train_sampler)

# Define data loader for test dataset
def _get_test_data_loader(test_batch_size, training_dir):
    logger.info("Get test data loader")
    
    test_set = torchvision.datasets.CIFAR10(root=training_dir, 
                                            train=False, 
                                            download=False, 
                                            transform=_get_transforms())
    
    return torch.utils.data.DataLoader(
        test_set,
        batch_size=test_batch_size,
        shuffle=True)

# Average gradients (only for multi-node CPU)
def _average_gradients(model):
    # Gradient averaging.
    ## We average the gradients and we dont have to do this if it is run on GPUs
    ## This function does a forward pass backward pass and use the gradients.
    size = float(dist.get_world_size())
    for param in model.parameters():
        dist.all_reduce(param.grad.data, op=dist.reduce_op.SUM)
        param.grad.data /= size

# Define training loop
def train(args):
    
    
    is_distributed = len(args.hosts) > 1 and args.backend is not None
    logger.debug("Distributed training - {}".format(is_distributed))
    
    # This script is for CPU only training, so define device as CPU
    device = torch.device("cpu")

    ## Here, if it is distributed, we look at how many instances (world instances), and rank is who 'i' am uniquely. 
    if is_distributed:
        world_size = len(args.hosts)
        os.environ["WORLD_SIZE"] = str(world_size)
        
        ## Each script knows who it is uniquely and use that information to do specific things you don't want everyone to do.
        host_rank = args.hosts.index(args.current_host)
        os.environ["RANK"] = str(host_rank)
        
        ## Here we initialize the init process group. This runs on all of the instances and helps to talk to each other (know that they are a part of the distrbuted system). 
        
        ## the backend used here is Gloo (for both), for GPU, we can use nickle
        dist.init_process_group(backend=args.backend, rank=host_rank, world_size=world_size)
        
        logger.info(
            "Initialized the distributed environment: '{}' backend on {} nodes. ".format(
                args.backend, dist.get_world_size()
            ))

    # Set the seed for generating random numbers
    torch.manual_seed(args.seed)

    train_loader = _get_train_data_loader(args.batch_size,     args.data_dir, is_distributed)
    test_loader  = _get_test_data_loader(args.test_batch_size, args.data_dir)

    logger.debug(
        "Processes {}/{} ({:.0f}%) of train data".format(
            len(train_loader.sampler),
            len(train_loader.dataset),
            100.0 * len(train_loader.sampler) / len(train_loader.dataset),
        )
    )

    logger.debug(
        "Processes {}/{} ({:.0f}%) of test data".format(
            len(test_loader.sampler),
            len(test_loader.dataset),
            100.0 * len(test_loader.sampler) / len(test_loader.dataset),
        )
    )

    model = Net().to(device)
    
    ## We just did the vanilla PyTorch training, and now here, we do the process of converting the model into the distributed training fashion.
    if is_distributed:
        model = torch.nn.parallel.DistributedDataParallel(model)

    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

    for epoch in range(1, args.epochs + 1):
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader, 1):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            if is_distributed:
                _average_gradients(model)
            optimizer.step()
            if batch_idx % args.log_interval == 0:
                logger.info(
                    "Train Epoch: {} [{}/{} ({:.0f}%)] Loss: {:.6f}".format(
                        epoch,
                        batch_idx * len(data),
                        len(train_loader.sampler),
                        100.0 * batch_idx / len(train_loader),
                        loss.item(),
                    )
                )
        test(model, test_loader, device)
    
    logger.info('Saving trained model only on rank 0')
    rank = os.getenv('RANK')
    if rank is not None:
        if int(rank) == 0:
            save_model(model, args.model_dir)
    else:
        save_model(model, args.model_dir)

def test(model, test_loader, device):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, size_average=False).item()  # sum up batch loss
            pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    logger.info(
        "Test set: Average loss: {:.4f}, Accuracy: {:.2f}\n".format(
            test_loss, correct / len(test_loader.dataset)
        )
    )

def save_model(model, model_dir):
    logger.info("Saving the model.")
    path = os.path.join(model_dir, "model.pth")
    torch.save(model.module.state_dict(), path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    ## Here the script should be able to accept some command line arguments from sagemaker. PyTorch arguments are the hyperparameters. These are the command line interface that sagemaker will pass when spinning up the container and then while passing to the script.

    # PyTorch environments
    parser.add_argument("--model-type",type=str,default='resnet18',
                        help="custom model or resnet18")
    parser.add_argument("--batch-size",type=int,default=64,
                        help="input batch size for training (default: 64)")
    parser.add_argument("--test-batch-size",type=int,default=1000,
                        help="input batch size for testing (default: 1000)")
    parser.add_argument("--epochs",type=int,default=10,
                        help="number of epochs to train (default: 10)")
    parser.add_argument("--lr", type=float, default=0.01,
                        help="learning rate (default: 0.01)")
    parser.add_argument("--momentum", type=float, default=0.5,
                        help="SGD momentum (default: 0.5)")
    parser.add_argument("--seed", type=int, default=1,
                        help="random seed (default: 1)")
    parser.add_argument("--log-interval",type=int,default=100,
                        help="how many batches to wait before logging training status")
    parser.add_argument("--backend",type=str,default='gloo',
                        help="backend for dist. training, this script only supports gloo")

    # SageMaker environment --> these are required to passed the data to s3 and then to the training script. it copied the data from s3 to container and so on too. 
    
    ## host and current host is only for the process of distributed training.
    parser.add_argument("--hosts", type=list, default=json.loads(os.environ["SM_HOSTS"]))
    parser.add_argument("--current-host", type=str, default=os.environ["SM_CURRENT_HOST"])
    parser.add_argument("--model-dir", type=str, default=os.environ["SM_MODEL_DIR"])
    parser.add_argument("--data-dir", type=str, default=os.environ["SM_CHANNEL_TRAIN"])

    train(parser.parse_args())
