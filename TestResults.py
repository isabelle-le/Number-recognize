# coding=utf-8
from __future__ import print_function
import torch
import torch.utils.data as Data
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.autograd import Variable
import cv2
import numpy as np

my_test_loader = Data.DataLoader(
    datasets.MNIST('./data', train=False, download=True,
                   transform=transforms.Compose([transforms.ToTensor(),
                                                 transforms.Normalize((0.1307,), (0.3081,))])),
    batch_size=1, shuffle=True
)
def my_show_test(my_model):

    for data, target in my_test_loader:
        # print('data.shape', data.numpy().shape)
        temp = data
        data, target = Variable(data, volatile=True), Variable(target)
        output = my_model(data)
        out = output.data[0].numpy()
        # print(out)
        print('perdict number is  : ', np.where(out == np.max(out))[0])
        im = cv2.resize((temp.numpy())[0].transpose([1, 2, 0]),(280,280),interpolation=cv2.INTER_CUBIC)
        cv2.imshow('lenet', im)
        cv2.waitKey(0)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x)

my_model = Net()
state_dict = torch.load('./lenet.pkl')['state_dict']
# from collections import OrderedDict
# new_state_dict = OrderedDict()
# for k, v in state_dict.items():
#     name = k[7:]
#     new_state_dict[name] = v
# my_model.load_state_dict(new_state_dict)
my_model.load_state_dict(state_dict)

my_show_test(my_model)
