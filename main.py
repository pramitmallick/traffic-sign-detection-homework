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
parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
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

def make_weights_for_balanced_classes(images, nclasses):                        
    count = [0] * nclasses                                                      
    for item in images:                                                         
        count[item[1]] += 1                                                     
    weight_per_class = [0.] * nclasses                                      
    N = float(sum(count))                                                   
    for i in range(nclasses):                                                   
        weight_per_class[i] = N/float(count[i])                                 
    weight = [0] * len(images)                                              
    for idx, val in enumerate(images):                                          
        weight[idx] = weight_per_class[val[1]]                                  
    return weight

### Data Initialization and Loading
from data import initialize_data, data_transforms # data.py in the same folder
initialize_data(args.data) # extracts the zip files, makes a validation set

# dataset_train = datasets.ImageFolder(args.data + '/train_images',
#                          transform=data_transforms)                                                                         
                                                                                
# # For unbalanced dataset we create a weighted sampler                       
# weights = make_weights_for_balanced_classes(dataset_train.imgs, len(dataset_train.classes))                                                                
# weights = torch.DoubleTensor(weights)                                       
# sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, len(weights))                     
                                                                                
# train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=args.batch_size, shuffle = False,                              
#                                                              sampler = sampler, num_workers=1)

# val_loader = torch.utils.data.DataLoader(
#     datasets.ImageFolder(args.data + '/val_images',
#                          transform=data_transforms),
#     batch_size=args.batch_size, shuffle=False, num_workers=1)

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

def train(epoch, convergencePlots, optimizer):
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
    return [train_loss, 100. * correct / len(train_loader.dataset)]

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
    return [validation_loss, 100. * correct / len(val_loader.dataset)]

convergencePlots = collections.defaultdict(list)
best_val_acc = 0

lr = args.lr
optimizer = optim.SGD(model.parameters(), lr=lr, momentum=args.momentum)
# optimizer = optim.Adagrad(model.parameters(), lr=args.lr)
val80 = False
val85 = False
val90 = False

for epoch in range(1, args.epochs + 1):
    [train_loss, train_acc] = train(epoch, convergencePlots, optimizer)
    [val_loss, val_acc] = validation(convergencePlots)
    print("Learning rate - ", lr)
    model_file = '/scratch/pm2758/cv_ass2/lr2_model_' + str(epoch) + '.pth'
    torch.save(model.state_dict(), model_file)
    convergencePlots['lr'].append(lr)
    if val_acc > 80 and not val80:
        lr /= 2
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=args.momentum)
        val80 = True
    if val_acc > 85 and not val85:
        lr /= 2
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=args.momentum)
        val85 = True
    if val_acc > 90 and not val90:
        lr /= 2
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=args.momentum)
        val90 = True
    
    # /scratch/pm2758/cv_ass2
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        model_file = 'model_stn_lr2.pth'
        torch.save(model.state_dict(), model_file)
        print('\nSaved model to ' + model_file + '. You can run `python evaluate.py ' + model_file + '` to generate the Kaggle formatted csv file')
        convergencePlots['best_val_acc'] = [epoch, best_val_acc]
    pickle.dump( convergencePlots, open( "convergencePlots_model_stn_lr2.p", "wb" ) )
