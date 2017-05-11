import cPickle, gzip
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import torchvision.transforms as transforms


def unpickle(file):
    fo = open(file, 'rb')
    dict_ = cPickle.load(fo)
    fo.close()
    return dict_

# Load CIFAR-10 data
train_x = []
train_y = []
for i in xrange(1, 5):
    dict_ = unpickle('../data/CIFAR-10/data_batch_' + str(i))
    if i == 1:
        train_x = np.array(dict_['data'])/255.0
        train_y = dict_['labels']
    else:
        train_x = np.concatenate((train_x, np.array(dict_['data'])/255.0), axis=0)
        train_y.extend(dict_['labels'])

train_y = np.array(train_y)
dict_ = unpickle('../data/CIFAR-10/data_batch_5')
valid_x = np.array(dict_['data'])/255.0
valid_y = np.array(dict_['labels'])
del dict_
train_x = np.dstack((train_x[:, :1024], train_x[:, 1024:2048], train_x[:, 2048:]))
train_x = np.reshape(train_x, [-1, 32, 32, 3])
train_x = np.transpose(train_x, (0, 3, 1, 2))
valid_x = np.dstack((valid_x[:, :1024], valid_x[:, 1024:2048], valid_x[:, 2048:]))
valid_x = np.reshape(valid_x, [-1, 32, 32, 3])
valid_x = np.transpose(valid_x, (0, 3, 1, 2))
train_x = torch.from_numpy(train_x).float().cuda()
valid_x = torch.from_numpy(valid_x).float().cuda()
train_y = torch.from_numpy(train_y).cuda()
valid_y = torch.from_numpy(valid_y).cuda()

width = 1


class Residual(nn.Module):
    def __init__(self, fan_in, fan_out, stride=1, w_init='xavier_normal'):
        super(Residual, self).__init__()
        self.fan_in, self.fan_out = fan_in, fan_out
        self.conv1 = nn.Conv2d(fan_in, fan_out, 3, stride=stride, padding=1)
        self.conv2 = nn.Conv2d(fan_out, fan_out, 3, padding=1)
        self.transform = nn.Conv2d(fan_out, fan_out, 3, padding=1)
        # self.expand_x = nn.Conv2d(fan_in, fan_out, 1)
        self.batch_norm1 = nn.BatchNorm2d(fan_in)
        self.batch_norm2 = nn.BatchNorm2d(fan_out)
        # Get weight initialization function
        w_initialization = getattr(nn.init, w_init)
        w_initialization(self.conv1.weight)
        nn.init.uniform(self.conv1.bias)
        w_initialization(self.conv2.weight)
        nn.init.uniform(self.conv2.bias)
        # w_initialization(self.expand_x.weight)
        # nn.init.uniform(self.expand_x.bias)

    def forward(self, x, downsample=False, train_mode=True):
        """
        Using mode='replicate' while padding cuz constant is not yet implemented for 5-D input
        :param x: input
        :param downsample: True for first conv layer in a 'stage'
        :param train_mode: Used for batch norm layer
        :return: output of the block
        """
        self.batch_norm1.training = train_mode
        self.batch_norm2.training = train_mode
        h = self.conv1(F.leaky_relu(self.batch_norm1(x)))
        h = self.conv2(F.leaky_relu(self.batch_norm2(h)))
        x_new = x
        if downsample:
            x_new = F.avg_pool2d(x_new, 2, 2)
        if self.fan_in != self.fan_out:
            x_new = F.pad(x_new.unsqueeze(0), (0, 0, 0, 0,(self.fan_out-self.fan_in)//2, (self.fan_out-self.fan_in)//2),
                          mode='replicate').squeeze(0)
        t = self.transform(h)
        return h * t + x_new * (1 - t), torch.sum(torch.squeeze(torch.max(torch.max(t, dim=2)[0], dim=3)[0]), dim=1)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.res11 = Residual(16, 16*width)
        self.res12 = Residual(16*width, 16*width)
        self.res13 = Residual(16*width, 16*width)
        self.res14 = Residual(16*width, 16*width)
        self.res21 = Residual(16*width, 32*width, stride=2)
        self.res22 = Residual(32*width, 32*width)
        self.res23 = Residual(32*width, 32*width)
        self.res24 = Residual(32*width, 32*width)
        self.res31 = Residual(32*width, 64*width, stride=2)
        self.res32 = Residual(64*width, 64*width)
        self.res33 = Residual(64*width, 64*width)
        self.res34 = Residual(64*width, 64*width)
        self.final = nn.Linear(64*width, 10)

    def forward(self, x, train_mode=True):
        net = self.conv1(x)
        t_sum = 0
        net, t = self.res11(net, train_mode=train_mode)
        t_sum += t
        net, t = self.res12(net, train_mode=train_mode)
        t_sum += t
        net, t = self.res13(net, train_mode=train_mode)
        t_sum += t
        net, t = self.res14(net, train_mode=train_mode)
        t_sum += t
        net, t = self.res21(net, train_mode=train_mode, downsample=True)
        t_sum += t
        net, t = self.res22(net, train_mode=train_mode)
        t_sum += t
        net, t = self.res23(net, train_mode=train_mode)
        t_sum += t
        net, t = self.res24(net, train_mode=train_mode)
        t_sum += t
        net, t = self.res31(net, train_mode=train_mode, downsample=True)
        t_sum += t
        net, t = self.res32(net, train_mode=train_mode)
        t_sum += t
        net, t = self.res33(net, train_mode=train_mode)
        t_sum += t
        net, t = self.res34(net, train_mode=train_mode)
        t_sum += t
        net = F.avg_pool2d(net, 8, 1)
        net = torch.squeeze(net)
        net = self.final(net)
        if train_mode:
            return net, torch.max(t_sum, dim=0)[0]
        return net


network = Net()
network = network.cuda()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(network.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4, nesterov=True)
transform = transforms.Compose([transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip()])

epochs = 250
batch_size = 128
print "Number of training examples : "+str(train_x.size(0))
for epoch in xrange(1, epochs + 1):

    sequence = torch.randperm(len(train_x)).cuda()
    train_x = train_x[sequence]
    train_y = train_y[sequence]

    if epoch > 150:
        optimizer = optim.SGD(network.parameters(), lr=0.0001, momentum=0.9, weight_decay=5e-4, nesterov=True)
    elif epoch > 60:
        optimizer = optim.SGD(network.parameters(), lr=0.001, momentum=0.9, weight_decay=5e-4, nesterov=True)
    cursor, t_cost_arr = 0, []
    while cursor < len(train_x):
        optimizer.zero_grad()
        outputs, t_cost = network(Variable(train_x[cursor:min(cursor + batch_size, len(train_x))]))
        t_cost_arr.append(t_cost.data[0][0])
        if epoch > 15:
            loss = criterion(outputs, Variable(train_y[cursor:min(cursor + batch_size, len(train_x))])) + 0.005 * t_cost
        else:
            loss = criterion(outputs, Variable(train_y[cursor:min(cursor + batch_size, len(train_x))]))
        loss.backward()
        optimizer.step()
        cursor += batch_size
    print round(min(t_cost_arr)), round(max(t_cost_arr))

    cursor, correct, total = 0, 0, 0
    while cursor < len(valid_x):
        outputs = network(Variable(valid_x[cursor:min(cursor + batch_size, len(valid_x))]), train_mode=False)
        labels = valid_y[cursor:min(cursor + batch_size, len(valid_x))]
        _, predicted = torch.max(outputs.data, 1)
        total += len(labels)
        correct += (predicted == labels).sum()
        cursor += batch_size

    print('For epoch %d \tAccuracy on valid set: %f %%' % (epoch, 100.0 * correct / total))
