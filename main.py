from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import pickle
import collections

# Training settings
parser = argparse.ArgumentParser(description='PyTorch GTSRB example')
parser.add_argument('--data', type=str, default='data', metavar='D',
                    help="folder where data is located. train_data.zip and test_data.zip need to be found in the folder")
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--epochs', type=int, default=100, metavar='N',
                    help='number of epochs to train (default: 1)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                    help='SGD momentum (default: 0.5)')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
args = parser.parse_args()

torch.manual_seed(args.seed)

cuda_available = False
if torch.cuda.is_available():
    cuda_available = True

### Data Initialization and Loading
from data import initialize_data, data_transforms # data.py in the same folder
initialize_data(args.data) # extracts the zip files, makes a validation set

train_loader = torch.utils.data.DataLoader(
    datasets.ImageFolder(args.data + '/train_images',
                         transform=data_transforms),
    batch_size=args.batch_size, shuffle=True, num_workers=1)
val_loader = torch.utils.data.DataLoader(
    datasets.ImageFolder(args.data + '/val_images',
                         transform=data_transforms),
    batch_size=args.batch_size, shuffle=False, num_workers=1)

### Neural Network and Optimizer
# We define neural net in model.py so that it can be reused by the evaluate.py script
from model import Net
model = Net()
if cuda_available:
    model.cuda()

# optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
optimizer = optim.Adagrad(model.parameters(), lr=args.lr)

def train(epoch, convergencePlots):
    model.train()
    sumLoss = 0
    numBatches = 0
    correct = 0
    train_loss = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        numBatches += 1
        if cuda_available:
            data = data.cuda()
            target = target.cuda()
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        # print("data", data.size())
        output = model(data)
        train_loss += F.cross_entropy(output, target, size_average=False).data[0] # sum up batch loss
        pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()
        # print("output", output.size())
        # print("target", target.size())
        # break
        # loss = F.nll_loss(output, target)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.data[0]))
        sumLoss += loss.data[0]

    train_loss /= len(train_loader.dataset)
    convergencePlots['training_avg_loss'].append(train_loss)
    convergencePlots['training_avg_acc'].append(100. * correct / len(train_loader.dataset))

def validation(convergencePlots):
    model.eval()
    validation_loss = 0
    correct = 0
    for data, target in val_loader:
        if cuda_available:
            data = data.cuda()
            target = target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)
        output = model(data)
        validation_loss += F.cross_entropy(output, target, size_average=False).data[0] # sum up batch loss
        pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    validation_loss /= len(val_loader.dataset)
    print('\nValidation set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        validation_loss, correct, len(val_loader.dataset),
        100. * correct / len(val_loader.dataset)))
    convergencePlots['validation_avg_loss'].append(validation_loss)
    convergencePlots['validation_avg_acc'].append(100. * correct / len(val_loader.dataset))

convergencePlots = collections.defaultdict(list)
for epoch in range(1, args.epochs + 1):
    train(epoch, convergencePlots)
    validation(convergencePlots)
    # model_file = 'model_' + str(epoch) + '.pth'
    model_file = 'model_latest_Adagrad.pth'
    torch.save(model.state_dict(), model_file)
    print('\nSaved model to ' + model_file + '. You can run `python evaluate.py ' + model_file + '` to generate the Kaggle formatted csv file')
    pickle.dump( convergencePlots, open( "convergencePlots_Adagrad.p", "wb" ) )
