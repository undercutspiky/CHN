import cPickle, gzip
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

# Load MNIST data
f = gzip.open('../../data/MNIST/mnist.pkl.gz', 'rb')
train_set, valid_set, test_set = cPickle.load(f)
f.close()
del test_set
train_x, train_y = train_set
valid_x, valid_y = valid_set
train_x = torch.from_numpy(train_x).cuda()
valid_x = torch.from_numpy(valid_x).cuda()
train_y = torch.from_numpy(train_y).cuda()
valid_y = torch.from_numpy(valid_y).cuda()


class Highway(nn.Module):
    def __init__(self, fan_in, fan_out, w_init='xavier_normal', b_init=-2.0):
        """
        Fully connected Highway Layer
        :param fan_in: Number of nodes in the previous layer
        :param fan_out: Number of nodes in this highway layer
        :param w_init: Any of the PyTorch's initialization methods
        :type w_init: String
        :param b_init: Initialization value for transform gates' bias
        :type b_init: Constant
        """
        super(Highway, self).__init__()
        self.fan_in = fan_in
        # Affine transformation layer and transform gates
        self.linear = nn.Linear(fan_in, fan_out)
        self.transform = nn.Linear(fan_in, fan_out)
        # self.batch_norm = nn.BatchNorm1d(fan_out)
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
        w_initialization(self.linear.weight)
        w_initialization(self.transform.weight)
        nn.init.constant(self.transform.bias, b_init)
        nn.init.uniform(self.linear.bias)

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
        # New linear layer
        linear = nn.Linear(self.fan_in, len(retain))
        linear.weight = torch.nn.Parameter(self.linear.weight[torch.cuda.LongTensor(retain)].data)
        linear.bias = torch.nn.Parameter(self.linear.bias[torch.cuda.LongTensor(retain)].data)
        self.linear = linear
        # New transform layer
        linear = nn.Linear(self.fan_in, len(retain))
        linear.weight = torch.nn.Parameter(self.transform.weight[torch.cuda.LongTensor(retain)].data)
        linear.bias = torch.nn.Parameter(self.transform.bias[torch.cuda.LongTensor(retain)].data)
        self.transform = linear
        # New batch normalization layer
        # bn = nn.BatchNorm1d(len(retain))
        # bn.weight = torch.nn.Parameter(self.batch_norm.weight[torch.cuda.LongTensor(retain)].data)
        # bn.bias = torch.nn.Parameter(self.batch_norm.bias[torch.cuda.LongTensor(retain)].data)
        # bn.running_mean = self.batch_norm.running_mean
        # bn.running_var = self.batch_norm.running_var
        # self.batch_norm = bn
        # Set self.order and reverse order
        self.order = self.order[torch.cuda.LongTensor(retain + remove)]
        for ii in xrange(len(self.order)):
            self.reverse_order[self.order[ii]] = ii
        self.reverse_order = torch.cuda.LongTensor(self.reverse_order)

    def forward(self, x, train_mode=True):
        if self.completely_pruned:
            return x, Variable(torch.zeros(x.size()).cuda())
        h = F.leaky_relu(self.linear(x))
        # Pad with zeros if layer is pruned
        h = F.pad(h.unsqueeze(0).unsqueeze(0), (0, x.size(1) - h.size(1), 0, 0)).squeeze(0).squeeze(0)
        t = F.sigmoid(self.transform(h))
        # self.batch_norm.training = train_mode
        # Pad with zeros if layer is pruned
        t = F.pad(t.unsqueeze(0).unsqueeze(0), (0, x.size(1) - t.size(1), 0, 0)).squeeze(0).squeeze(0)
        out = h * t + (x.t()[self.order].t() * (1 - t))
        return out.t()[self.reverse_order].t(), t


class Net(nn.Module):
    def __init__(self, fan_in=784, fan_out=128):
        super(Net, self).__init__()
        self.linear = nn.Linear(fan_in, fan_out)
        self.highway_layers = nn.ModuleList()
        self.final = nn.Linear(fan_out, 10)
        for i in xrange(6):
            self.highway_layers.append(Highway(fan_out, fan_out).cuda())

    def forward(self, x, train_mode=True, get_t=False):
        net = F.leaky_relu(self.linear(x))
        temp, t_sum = None, 0
        for layer in self.highway_layers:
            net, t = layer(net, train_mode)
            t_sum += torch.sum(t, dim=1)
            if get_t:
                if temp is None:
                    temp = np.expand_dims(t.data.cpu().numpy(), axis=1)
                else:
                    temp = np.append(temp, np.expand_dims(t.data.cpu().numpy(), axis=1), axis=1)
        net = self.final(net)
        if get_t:
            return net, temp
        if train_mode:
            return net, torch.max(t_sum, dim=0)[0]
        return net


def loss(y, targets):
    temp = F.softmax(y)
    l = [-torch.log(temp[i][targets[i].data[0]]) for i in range(y.size(0))]
    return F.cross_entropy(y, targets), l


network = Net()
network = network.cuda()
print network
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(network.parameters(), lr=0.1, momentum=0.7, weight_decay=0.0001)

epochs = 150
batch_size = 128
prune_at = [20, 50, 70, 90, 110, 130]

for epoch in xrange(1, epochs + 1):

    sequence = torch.randperm(len(train_x)).cuda()
    train_x = train_x[sequence]
    train_y = train_y[sequence]

    for iii in xrange(len(network.highway_layers)):
        network.highway_layers[iii].completely_pruned = True

    if epoch > 50:
        optimizer = optim.SGD(network.parameters(), lr=0.01, momentum=0.7, weight_decay=0.0001)
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
                if max_values[i][j] < 0.2:
                    rem.append(j)
                else:
                    ret.append(j)
            network.highway_layers[i].prune(ret, rem)
            if not network.highway_layers[i].completely_pruned:
                print network.highway_layers[i]

    cursor, t_cost_arr = 0, []
    while cursor < len(train_x):
        optimizer.zero_grad()
        outputs, t_cost = network(Variable(train_x[cursor:min(cursor + batch_size, len(train_x))]))
        t_cost_arr.append(t_cost.data[0][0])
        if epoch > 10:
            loss = criterion(outputs, Variable(train_y[cursor:min(cursor + batch_size, len(train_x))])) + 0.10 * t_cost
        else:
            loss = criterion(outputs, Variable(train_y[cursor:min(cursor + batch_size, len(train_x))]))
        loss.backward()
        nn.utils.clip_grad_norm(network.parameters(), 1.0)
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
