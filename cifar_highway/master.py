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
        # The number of the node in the previous layer to which output of each node in this layer is added
        # Look at forward function for more clarity
        self.order = range(fan_out)
        self.reverse_order = [0] * fan_out
        self.completely_pruned = False
        for ii in xrange(fan_out):
            self.reverse_order[self.order[ii]] = ii
        self.order = torch.cuda.LongTensor(self.order)
        self.reverse_order = torch.cuda.LongTensor(self.reverse_order)
        # Get weight initialization function
        w_initialization = getattr(nn.init, w_init)
        w_initialization(self.conv1.weight)
        nn.init.uniform(self.conv1.bias)
        w_initialization(self.conv2.weight)
        nn.init.uniform(self.conv2.bias)
        # w_initialization(self.expand_x.weight)
        # nn.init.uniform(self.expand_x.bias)

    def prune(self, retain=[], remove=[]):
        """
        Create new layers and copy the parameters of the selected nodes.
        :param retain: list of nodes to retain
        :type retain: python List
        :param remove: list of nodes to remove (helps in setting self.order)
        :type remove: python List
        """
        if len(retain) == 0 or self.completely_pruned:
            self.completely_pruned = True
            print 'Completely Pruned !'
            return
        # New conv layer
        conv = nn.Conv2d(self.fan_out, len(retain), 3, padding=1)
        conv.weight = torch.nn.Parameter(self.conv2.weight[torch.cuda.LongTensor(retain)].data)
        conv.bias = torch.nn.Parameter(self.conv2.bias[torch.cuda.LongTensor(retain)].data)
        self.conv2 = conv
        # New transform layer
        conv = nn.Conv2d(self.fan_out, len(retain), 3, padding=1)
        conv.weight = torch.nn.Parameter(self.transform.weight[torch.cuda.LongTensor(retain)].data)
        conv.bias = torch.nn.Parameter(self.transform.bias[torch.cuda.LongTensor(retain)].data)
        self.transform = conv
        # New batch normalization layer
        bn = nn.BatchNorm2d(len(retain))
        bn.weight = torch.nn.Parameter(self.batch_norm2.weight[torch.cuda.LongTensor(retain)].data)
        bn.bias = torch.nn.Parameter(self.batch_norm2.bias[torch.cuda.LongTensor(retain)].data)
        bn.running_mean = self.batch_norm2.running_mean
        bn.running_var = self.batch_norm2.running_var
        self.batch_norm2 = bn
        # Set self.order and reverse order
        self.order = self.order[torch.cuda.LongTensor(retain + remove)]
        for ii in xrange(len(self.order)):
            self.reverse_order[self.order[ii]] = ii
        self.reverse_order = torch.cuda.LongTensor(self.reverse_order)

    def forward(self, x, downsample=False, train_mode=True):
        """
        Using mode='replicate' while padding cuz constant is not yet implemented for 5-D input
        :param x: input
        :param downsample: True for first conv layer in a 'stage'
        :param train_mode: Used for batch norm layer
        :return: output of the block
        """
        if self.completely_pruned:
            return x, Variable(torch.zeros(x.size()).cuda())
        self.batch_norm1.training = train_mode
        self.batch_norm2.training = train_mode
        h = self.conv1(F.leaky_relu(self.batch_norm1(x)))
        h = self.conv2(F.leaky_relu(self.batch_norm2(h)))
        t = F.sigmoid(self.transform(h))
        x_new = x
        if downsample:
            x_new = F.avg_pool2d(x_new, 2, 2)
        if self.fan_in != self.fan_out:
            x_new = F.pad(x_new.unsqueeze(0), (0, 0, 0, 0, (self.fan_out-self.fan_in)//2, (self.fan_out-self.fan_in)//2)
                          , mode='replicate').squeeze(0)
        # Padding - to match dimensions of pruned layer and x_new
        h = torch.squeeze(F.pad(h.unsqueeze(0), (0, 0, 0, 0, 0, x_new.size(1) - h.size(1)), mode='replicate'))
        t = torch.squeeze(F.pad(t.unsqueeze(0), (0, 0, 0, 0, 0, x_new.size(1) - t.size(1)), mode='replicate'))
        # This is where self.order comes in use after the layer has been pruned
        out = h * t + (x_new.permute(1, 0, 2, 3)[self.order].permute(1, 0, 2, 3) * (1 - t))
        out = out.permute(1, 0, 2, 3)[self.reverse_order].permute(1, 0, 2, 3)
        return out, torch.sum(torch.squeeze(torch.max(torch.max(t, dim=2)[0], dim=3)[0]), dim=1)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.highway_layers = []
        self.highway_layers.append(Residual(16, 16*width))
        for i in xrange(3):
            self.highway_layers.append(Residual(16*width, 16*width))
        self.highway_layers.append(Residual(16*width, 32*width, stride=2))
        for i in xrange(3):
            self.highway_layers.append(Residual(32*width, 32*width))
        self.highway_layers.append(Residual(32*width, 64*width, stride=2))
        for i in xrange(3):
            self.highway_layers.append(Residual(64*width, 64*width))
        self.final = nn.Linear(64*width, 10)
        self.pruned = False

    def forward(self, x, train_mode=True, get_t=False):
        net = self.conv1(x)
        t_sum, temp = 0, None
        for layer in self.highway_layers:
            net, t = layer(net, train_mode, downsample=(layer.fan_in < layer.fan_out))
            t_sum += t
            if get_t:
                if temp is None:
                    temp = np.expand_dims(t.data.cpu().numpy(), axis=1)
                else:
                    temp = np.append(temp, np.expand_dims(t.data.cpu().numpy(), axis=1), axis=1)
        net = F.avg_pool2d(net, 8, 1)
        net = torch.squeeze(net)
        net = self.final(net)
        if get_t:
            return net, temp
        if train_mode:
            return net, torch.max(t_sum, dim=0)[0]
        return net


network = Net()
network = network.cuda()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(network.parameters(), lr=0.05, momentum=0.9, weight_decay=5e-4, nesterov=True)
transform = transforms.Compose([transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip()])

epochs = 300
batch_size = 128
print "Number of training examples : "+str(train_x.size(0))
prune_at = [90, 130, 200, 250]

for epoch in xrange(1, epochs + 1):

    sequence = torch.randperm(len(train_x)).cuda()
    train_x = train_x[sequence]
    train_y = train_y[sequence]

    if epoch > 150:
        optimizer = optim.SGD(network.parameters(), lr=0.0005, momentum=0.9, weight_decay=5e-4, nesterov=True)
    elif epoch > 60:
        optimizer = optim.SGD(network.parameters(), lr=0.005, momentum=0.9, weight_decay=5e-4, nesterov=True)

    if epoch == 1:
        for l in network.highway_layers:
            l.prune(range(10), range(10, l.fan_out-10))
    if epoch == 2:
        network.highway_layers[4].completely_pruned = True

    if epoch in prune_at:
        cursor, t_values = 0, 0
        while cursor < len(train_x):
            outputs, t_batch = network(Variable(train_x[cursor:min(cursor + batch_size, len(train_x))]), get_t=True)
            if cursor == 0:
                t_values = t_batch
            else:
                t_values = np.append(t_values, t_batch, axis=0)
            cursor += batch_size
        max_values = np.max(t_values, axis=0)
        for i in xrange(len(max_values)):
            ret, rem = [], []
            for j in xrange(len(max_values[i])):
                if max_values[i][j] < 0.1:
                    rem.append(j)
                else:
                    ret.append(j)
            network.highway_layers[i].prune(ret, rem)
            if not network.highway_layers[i].completely_pruned:
                print network.highway_layers[i].conv2

    cursor, t_cost_arr = 0, []
    while cursor < len(train_x):
        optimizer.zero_grad()
        outputs, t_cost = network(Variable(train_x[cursor:min(cursor + batch_size, len(train_x))]))
        if not network.pruned:
            t_cost_arr.append(t_cost.data[0][0])
        if epoch > 20:
            loss = criterion(outputs, Variable(train_y[cursor:min(cursor + batch_size, len(train_x))])) + 3e-3 * t_cost
        else:
            loss = criterion(outputs, Variable(train_y[cursor:min(cursor + batch_size, len(train_x))]))
        loss.backward()
        nn.utils.clip_grad_norm(network.parameters(), 1.0)
        optimizer.step()
        cursor += batch_size
    if not network.pruned:
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
