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
    dict_ = unpickle('../../data/CIFAR-10/data_batch_' + str(i))
    if i == 1:
        train_x = np.array(dict_['data'])/255.0
        train_y = dict_['labels']
    else:
        train_x = np.concatenate((train_x, np.array(dict_['data'])/255.0), axis=0)
        train_y.extend(dict_['labels'])

train_y = np.array(train_y)
dict_ = unpickle('../../data/CIFAR-10/data_batch_5')
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

width = 4


class Residual(nn.Module):
    def __init__(self, fan_in, fan_out, stride=1, w_init='xavier_normal'):
        super(Residual, self).__init__()
        self.fan_in, self.fan_out, self.stride = fan_in, fan_out, stride
        self.conv = nn.Conv2d(fan_in, fan_out, 3, stride=stride, padding=1)
        self.transform = nn.Conv2d(fan_out, fan_out, 3, padding=1)
        # self.expand_x = nn.Conv2d(fan_in, fan_out, 1)
        self.batch_norm = nn.BatchNorm2d(fan_in)
        # The number of the node in the previous layer to which output of each node in this layer is added
        # Look at forward function for more clarity
        self.order = range(fan_out)
        self.reverse_order = [0] * fan_out
        self.completely_pruned = False
        self.pruned = False
        for ii in xrange(fan_out):
            self.reverse_order[self.order[ii]] = ii
        self.order = torch.cuda.LongTensor(self.order)
        self.reverse_order = torch.cuda.LongTensor(self.reverse_order)
        self.downsample = self.fan_in < self.fan_out
        self.mask_x = None
        self.mask_h = None
        # Get weight initialization function
        w_initialization = getattr(nn.init, w_init)
        w_initialization(self.conv.weight)
        nn.init.uniform(self.conv.bias)
        w_initialization(self.transform.weight)
        nn.init.constant(self.transform.bias, -2.0)

    def forward(self, x, train_mode=True):
        """
        Using mode='replicate' while padding cuz constant is not yet implemented for 5-D input
        :param x: input
        :param train_mode: Used for batch norm layer
        :return: output of the block
        """
        if self.completely_pruned:
            x_new = x
            if self.downsample:
                x_new = F.avg_pool2d(x_new, 2, 2)
            if self.fan_in != self.fan_out:
                x_new = F.pad(x_new.unsqueeze(0), (0, 0, 0, 0, (self.fan_out-self.fan_in)//2,
                                                   (self.fan_out-self.fan_in)//2), mode='replicate').squeeze(0)
                # To ameliorate mode='replicate'
                if self.mask_x is None or self.mask_x.size() != x_new.size():
                    self.mask_x = torch.zeros(x_new.size())
                    maps_i = range((self.fan_out - self.fan_in) // 2, x_new.size(1) - (self.fan_out - self.fan_in) // 2)
                    maps_i_size = self.mask_x.permute(1, 0, 2, 3)[torch.LongTensor(maps_i)].size()
                    self.mask_x.permute(1, 0, 2, 3)[torch.LongTensor(maps_i)] = torch.ones(maps_i_size)
                x_new = x_new * Variable(self.mask_x).cuda()
            return x_new, Variable(torch.zeros(x_new.size(0), x_new.size(1)).cuda())
        self.batch_norm.training = train_mode
        h = self.conv(F.relu6(self.batch_norm(x)))
        t = F.sigmoid(self.transform(h))
        x_new = x
        if self.downsample:
            x_new = F.avg_pool2d(x_new, 2, 2)
        if self.fan_in != self.fan_out:
            x_new = F.pad(x_new.unsqueeze(0), (0, 0, 0, 0, (self.fan_out-self.fan_in)//2, (self.fan_out-self.fan_in)//2)
                          , mode='replicate').squeeze(0)
            # To ameliorate mode='replicate'
            if self.mask_x is None or self.mask_x.size() != x_new.size():
                self.mask_x = torch.zeros(x_new.size())
                maps_i = range((self.fan_out - self.fan_in) // 2, x_new.size(1) - (self.fan_out - self.fan_in) // 2)
                maps_i_size = self.mask_x.permute(1, 0, 2, 3)[torch.LongTensor(maps_i)].size()
                self.mask_x.permute(1, 0, 2, 3)[torch.LongTensor(maps_i)] = torch.ones(maps_i_size)
            x_new = x_new * Variable(self.mask_x).cuda()
        # Number of feature maps in h
        maps_i = range(h.size(1))
        # Padding - to match dimensions of pruned layer and x_new
        h = torch.squeeze(F.pad(h.unsqueeze(0), (0, 0, 0, 0, 0, x_new.size(1) - h.size(1)), mode='replicate'))
        # To ameliorate mode='replicate'
        if self.mask_h is None or self.mask_h.size() != h.size():
            self.mask_h = torch.zeros(h.size())
            maps_i_size = self.mask_h.permute(1, 0, 2, 3)[torch.LongTensor(maps_i)].size()
            self.mask_h.permute(1, 0, 2, 3)[torch.LongTensor(maps_i)] = torch.ones(maps_i_size)
        h = h * Variable(self.mask_h, requires_grad=False).cuda()
        t = torch.squeeze(F.pad(t.unsqueeze(0), (0, 0, 0, 0, 0, x_new.size(1) - t.size(1)), mode='replicate'))
        # To ameliorate mode='replicate'
        t = t * Variable(self.mask_h, requires_grad=False).cuda()
        # This is where self.order comes in use after the layer has been pruned
        out = h * t #+ (x_new.permute(1, 0, 2, 3)[self.order].permute(1, 0, 2, 3) * (1 - t))
        out += (x_new.permute(1, 0, 2, 3)[self.order].permute(1, 0, 2, 3) * (1 - t))
        out = out.permute(1, 0, 2, 3)[self.reverse_order].permute(1, 0, 2, 3)
        return out, torch.squeeze(torch.max(torch.max(t, dim=2)[0], dim=3)[0])

    def prune(self, retain=[], remove=[]):
        """
        Create new layers and copy the parameters of the selected nodes.
        :param retain: list of nodes to retain
        :type retain: python List
        :param remove: list of nodes to remove (helps in setting self.order)
        :type remove: python List
        """
        if len(remove) > 0:
            self.pruned = True
        if len(retain) == 0 or self.completely_pruned:
            self.completely_pruned = True
            print 'Completely Pruned !'
            return
        # New conv layer
        conv = nn.Conv2d(self.fan_in, len(retain), 3, stride=self.stride, padding=1)
        conv.weight = torch.nn.Parameter(self.conv.weight[torch.cuda.LongTensor(retain)].data)
        conv.bias = torch.nn.Parameter(self.conv.bias[torch.cuda.LongTensor(retain)].data)
        self.conv = conv
        # New transform layer
        conv = nn.Conv2d(len(retain), len(retain), 3, padding=1)
        # Transfer weights to cpu then to cuda to avoid RuntimeError: cuDNN requires contiguous weight tensor
        conv.weight = torch.nn.Parameter(self.transform.weight[torch.cuda.LongTensor(retain)].permute(1, 0, 2, 3)
                                         [torch.cuda.LongTensor(retain)].permute(1, 0, 2, 3).data.cpu().cuda())
        conv.bias = torch.nn.Parameter(self.transform.bias[torch.cuda.LongTensor(retain)].data)
        self.transform = conv
        # Set self.order and reverse order
        self.order = self.order[torch.cuda.LongTensor(retain + remove)]
        for ii in xrange(len(self.order)):
            self.reverse_order[self.order[ii]] = ii
        self.reverse_order = torch.cuda.LongTensor(self.reverse_order)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.highway_layers = nn.ModuleList()
        self.highway_layers.append(Residual(16, 16*width).cuda())
        self.highway_layers[0].downsample = False
        for _ in xrange(5):
            self.highway_layers.append(Residual(16*width, 16*width).cuda())
        self.highway_layers.append(Residual(16*width, 32*width, stride=2).cuda())
        for _ in xrange(5):
            self.highway_layers.append(Residual(32*width, 32*width).cuda())
        self.highway_layers.append(Residual(32*width, 64*width, stride=2).cuda())
        for _ in xrange(5):
            self.highway_layers.append(Residual(64*width, 64*width).cuda())
        self.final = nn.Linear(64*width, 10)

    def forward(self, x, train_mode=True, get_t=False):
        net = self.conv1(x)
        t_sum, temp1, temp2, temp3 = 0, None, None, None
        for iii in xrange(len(self.highway_layers)):
            net, t = self.highway_layers[iii](net, train_mode=train_mode)
            t_sum += torch.sum(t, dim=1)
            if get_t:
                if iii < 6:
                    temp1 = self.get_t_arr(temp1, t)
                elif iii < 12:
                    temp2 = self.get_t_arr(temp2, t)
                else:
                    temp3 = self.get_t_arr(temp3, t)
        net = F.avg_pool2d(net, 8, 1)
        net = torch.squeeze(net)
        net = self.final(net)
        if get_t:
            return net, [temp1, temp2, temp3]
        if train_mode:
            return net, torch.max(t_sum, dim=0)[0]
        return net

    def get_t_arr(self, temp, t):
        if temp is None:
            temp = np.expand_dims(t.data.cpu().numpy(), axis=1)
        else:
            temp = np.append(temp, np.expand_dims(t.data.cpu().numpy(), axis=1), axis=1)
        return temp


def train(tco=0.0):
    cursor, t_cost_arr = 0, []
    while cursor + batch_size <= len(train_x):  # So that masks are created only once -ignore last batch of smaller size
        optimizer.zero_grad()
        outputs, t_cost = network(Variable(train_x[cursor:min(cursor + batch_size, len(train_x))]))
        t_cost_arr.append(t_cost.data[0][0])
        loss = criterion(outputs, Variable(train_y[cursor:min(cursor + batch_size, len(train_x))])) + tco * t_cost
        loss.backward()
        nn.utils.clip_grad_norm(network.parameters(), 1.0)
        optimizer.step()
        cursor += batch_size

    print round(min(t_cost_arr)), round(max(t_cost_arr))


def validate():
    cursor, correct, total = 0, 0, 0
    while cursor < len(valid_x):
        outputs = network(Variable(valid_x[cursor:min(cursor + batch_size, len(valid_x))]), train_mode=False)
        labels = valid_y[cursor:min(cursor + batch_size, len(valid_x))]
        _, predicted = torch.max(outputs.data, 1)
        total += len(labels)
        correct += (predicted == labels).sum()
        cursor += batch_size

    return 100.0 * correct / total


def save_state(state_name):
    torch.save(network.state_dict(), './model-' + state_name + '.pth')
    torch.save(optimizer.state_dict(), './optimizer-' + state_name + '.pth')


def restore_state(state_name):
    network.load_state_dict(torch.load('./model-' + state_name + '.pth'))
    #optimizer.load_state_dict(torch.load('./optimizer-' + state_name + '.pth'))

network = Net()
network = network.cuda()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(network.parameters(), lr=0.0005, momentum=0.9, weight_decay=5e-4, nesterov=True)
transform = transforms.Compose([transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip()])

epochs = 20
batch_size = 128
print "Number of training examples : "+str(train_x.size(0))
prune_at = [1]
tc = 3e-3/width

restore_state('140')

for epoch in xrange(1, epochs + 1):

    sequence = torch.randperm(len(train_x)).cuda()
    train_x = train_x[sequence]
    train_y = train_y[sequence]

    if epoch in prune_at:
        '''cursor, t_values1, t_values2, t_values3 = 0, 0, 0, 0
        while cursor < len(valid_x):
            outputs, t_batch = network(Variable(valid_x[cursor:min(cursor + batch_size, len(valid_x))]), get_t=True)
            if cursor == 0:
                t_values1, t_values2, t_values3 = t_batch
            else:
                t_values1 = np.append(t_values1, t_batch[0], axis=0)
                t_values2 = np.append(t_values2, t_batch[1], axis=0)
                t_values3 = np.append(t_values3, t_batch[2], axis=0)
            cursor += batch_size
        max_values1 = np.max(t_values1, axis=0)
        max_values2 = np.max(t_values2, axis=0)
        max_values3 = np.max(t_values3, axis=0)
        np.save('max_values1', max_values1)
        np.save('max_values2', max_values2)
        np.save('max_values3', max_values3)'''
        max_values1 = np.load('max_values1.npy')
        max_values2 = np.load('max_values2.npy')
        max_values3 = np.load('max_values3.npy')
        threshold = 0.005
        layers_to_remove = []

        for param in network.parameters():
            param.requires_grad = False
        for param in network.final.parameters():
            param.requires_grad = True
        for i in reversed(range(len(network.highway_layers))):
            if i in [0, 6, 12]:
                continue
            ret, rem = [], []
            if i < 6:
                max_values = max_values1[i % 6]
                #threshold = 0.015
            elif i < 12:
                max_values = max_values2[i % 6]
                #threshold = 0.025
            else:
                max_values = max_values3[i % 6]
                #threshold = 0.035
            for j in xrange(len(max_values)):
                if max_values[j] < threshold:
                    rem.append(j)
                else:
                    ret.append(j)
            network.highway_layers[i].prune(ret, rem)
            if not network.highway_layers[i].completely_pruned:
                print network.highway_layers[i].conv
            else:
                layers_to_remove.append(i)

            print('Accuracy on valid set after pruning layer %d: %f %%' % (i, validate()))

            for param in network.highway_layers[i].parameters():
                param.requires_grad = True
            if i > 12:
                train()
                print('Accuracy after pruning and training for 1 epoch layer %d onwards: %f %%' % (i, validate()))
            elif i > 6:
                train()
                print('Accuracy after pruning and training for 1 epoch layer %d onwards: %f %%' % (i, validate()))
                train()
                print('Accuracy after pruning and training for 2 epochs layer %d onwards: %f %%' % (i, validate()))
            else:
                train()
                print('Accuracy after pruning and training for 1 epoch layer %d onwards: %f %%' % (i, validate()))
                train()
                print('Accuracy after pruning and training for 2 epochs layer %d onwards: %f %%' % (i, validate()))
                train()
                print('Accuracy after pruning and training for 3 epochs layer %d onwards: %f %%' % (i, validate()))
        for param in network.parameters():
            param.requires_grad = True
        layers = nn.ModuleList()
        # Don't remove dimensionality changing layers using the loop below !
        for i, l in enumerate(network.highway_layers):
            if i not in layers_to_remove:
                layers.append(network.highway_layers[i])
        network.highway_layers = layers
        optimizer = optim.SGD(network.parameters(), lr=0.005, momentum=0.9, weight_decay=5e-4, nesterov=True)

    if epoch == 10:
        optimizer = optim.SGD(network.parameters(), lr=0.0005, momentum=0.9, weight_decay=5e-4, nesterov=True)
    train()

    cursor, correct, total = 0, 0, 0

    print('For epoch %d \tAccuracy on valid set: %f %%' % (epoch, validate()))
save_state('pruned')
