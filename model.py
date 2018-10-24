import torch
import torch.nn as nn
import torch.nn.functional as F

nclasses = 43 # GTSRB as 43 classes

# class Net(nn.Module):
#     def __init__(self):
#         super(Net, self).__init__()
#         self.conv1 = nn.Conv2d(3, 10, kernel_size=5)
#         self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
#         self.conv2_drop = nn.Dropout2d()
#         self.fc1 = nn.Linear(500, 50)
#         self.fc2 = nn.Linear(50, nclasses)

#     def forward(self, x):
#         x = F.relu(F.max_pool2d(self.conv1(x), 2))
#         x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
#         x = x.view(-1, 500)
#         x = F.relu(self.fc1(x))
#         x = F.dropout(x, training=self.training)
#         x = self.fc2(x)
#         return F.log_softmax(x)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(500, 50)
        self.fc2 = nn.Linear(50, nclasses)

        # self.conv1 = nn.Conv2d(3, 70, kernel_size=5, padding=2)
        # self.conv2 = nn.Conv2d(70, 110, kernel_size=3, padding=1)
        # self.conv3 = nn.Conv2d(110, 180, kernel_size=3, padding=1)
        # self.conv1_1 = nn.BatchNorm2d(70)
        # self.conv2_1 = nn.BatchNorm2d(110)
        # self.conv3_1 = nn.BatchNorm2d(180)
        # self.conv1_drop = nn.Dropout2d()
        # self.conv2_drop = nn.Dropout2d()
        # self.conv3_drop = nn.Dropout2d()
        # self.fc1 = nn.Linear(4*4*180, 200)
        # self.fbn = nn.BatchNorm1d(200)
        # self.fc2 = nn.Linear(200, nclasses)

        # Spatial transformer localization-network
        self.localization = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=7),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(8, 10, kernel_size=5),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True)
        )

        # Regressor for the 3 * 2 affine matrix
        self.fc_loc = nn.Sequential(
            nn.Linear(10 * 4 * 4, 32),
            nn.ReLU(True),
            nn.Linear(32, 3 * 2)
        )

        # Initialize the weights/bias with identity transformation
        self.fc_loc[2].weight.data.zero_()
        # self.fc_loc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))
        self.fc_loc[2].bias.data.copy_(torch.FloatTensor([1, 0, 0, 0, 1, 0]))

    # Spatial transformer network forward function
    def stn(self, x):
        xs = self.localization(x)
        xs = xs.view(-1, 10 * 4 * 4)
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)

        grid = F.affine_grid(theta, x.size())
        x = F.grid_sample(x, grid)

        return x

    def forward(self, x):
        # x = self.stn(x)
        # x = self.conv1_1(F.relu(F.max_pool2d(self.conv1_drop(self.conv1(x)), 2)))
        # x = self.conv2_1(F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2)))
        # x = self.conv3_1(F.relu(F.max_pool2d(self.conv3_drop(self.conv3(x)), 2)))
        # x = x.view(-1, 4*4*180)
        # x = F.relu(self.fbn(self.fc1(x)))

        x = self.stn(x)
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 500)
        x = F.relu(self.fc1(x))
        
        
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        x = F.log_softmax(x, dim=0)
        return x