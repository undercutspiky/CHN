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
sequence = np.random.choice(len(train_x), size=len(train_x), replace=False)
train_x = train_x[sequence]
train_y = train_y[sequence]
valid_x, valid_y = valid_set
train_x = torch.from_numpy(train_x).cuda()
valid_x = torch.from_numpy(valid_x).cuda()
train_y = torch.from_numpy(train_y).cuda()
valid_y = torch.from_numpy(valid_y).cuda()


class Highway(nn.Module):
    def __init__(self, fan_in, fan_out, w_init='kaiming_normal', b_init=-2.0):
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
        self.batch_norm = nn.BatchNorm1d(fan_out)
        # The number of the node in the previous layer to which output of each node in this layer is added
        # Look at forward function for more clarity
        self.order = range(fan_out)
        # Get weight initialization function
        w_initialization = getattr(nn.init, w_init)
        w_initialization(self.linear.weight)
        w_initialization(self.transform.weight)
        nn.init.constant(self.transform.bias, b_init)
        nn.init.uniform(self.linear.bias)

    def prune(self, nodes):
        """
        Create new layers and copy the parameters of the selected nodes
        :param nodes: list of nodes to retain
        :type nodes: python List
        """
        self.order = nodes
        # New linear layer
        linear = nn.Linear(self.fan_in, len(nodes))
        linear.weight = torch.nn.Parameter(self.linear.weight[torch.LongTensor(nodes)].data)
        linear.bias = torch.nn.Parameter(self.linear.bias[torch.LongTensor(nodes)].data)
        self.linear = linear
        # New transform layer
        linear = nn.Linear(self.fan_in, len(nodes))
        linear.weight = torch.nn.Parameter(self.transform.weight[torch.LongTensor(nodes)].data)
        linear.bias = torch.nn.Parameter(self.transform.bias[torch.LongTensor(nodes)].data)
        self.transform = linear
        # New batch normalization layer
        bn = nn.BatchNorm1d(len(nodes))
        bn.weight = torch.nn.Parameter(self.batch_norm.weight[torch.LongTensor(nodes)].data)
        bn.bias = torch.nn.Parameter(self.batch_norm.bias[torch.LongTensor(nodes)].data)
        bn.running_mean = self.batch_norm.running_weight
        bn.running_var = self.batch_norm.running_var
        self.batch_norm = bn

    def forward(self, x, train_mode=True):
        h = F.leaky_relu(self.linear(x))
        t = F.sigmoid(self.transform(x))
        self.batch_norm.training = train_mode
        for i in xrange(len(self.order)):
            x[self.order[i]] = (1 - t[i]) * x[self.order[i]] + h[i] * t[i]
        return self.batch_norm(x), t


class Net(nn.Module):
    def __init__(self, fan_in=784, fan_out=250):
        super(Net, self).__init__()
        self.linear = nn.Linear(fan_in, fan_out)
        self.highway_layers = []
        self.final = nn.Linear(fan_out, 10)
        for i in xrange(5):
            self.highway_layers.append(Highway(fan_out, fan_out).cuda())

    def forward(self, x, train_mode=True, get_t=False):
        net = F.leaky_relu(self.linear(x))
        temp, t_sum = None, 0
        for layer in self.highway_layers:
            net, t = layer(net, train_mode)
            # Sum of all transform gate for all nodes after taking max value in the batch for each node
            t_sum = torch.sum(torch.max(t, dim=0)[0])
            if get_t:
                if temp is None:
                    temp = np.expand_dims(t.numpy(), axis=1)
                else:
                    temp = np.append(temp, np.expand_dims(t.numpy(), axis=1), axis=1)
        net = self.final(net)
        if get_t:
            return net, temp.numpy()
        if train_mode:
            return net, t_sum
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

epochs = 100
batch_size = 128

for epoch in xrange(1, epochs+1):

    if epoch > 30:
        optimizer = optim.SGD(network.parameters(), lr=0.01, momentum=0.7, weight_decay=0.0001)
    if epoch == 50 or epoch == 3:
        cursor, t_values = 0, 0
        while cursor < len(train_x):
            outputs, t_batch = network(Variable(train_x[cursor:min(cursor + batch_size, len(train_x))]), get_t=True)
            if cursor == 0:
                t_values = t_batch
            else:
                t_values = np.append(t_values, t_batch, axis=0)
        print t_values.shape
    cursor = 0
    while cursor < len(train_x):
        optimizer.zero_grad()
        outputs, t_cost = network(Variable(train_x[cursor:min(cursor+batch_size, len(train_x))]))
        if epoch > 20:
            loss = criterion(outputs, Variable(train_y[cursor:min(cursor+batch_size, len(train_x))])) + 0.01*t_cost
        else:
            loss = criterion(outputs, Variable(train_y[cursor:min(cursor+batch_size, len(train_x))]))
        loss.backward()
        optimizer.step()
        cursor += batch_size
    
    cursor = 0
    correct = 0
    total = 0
    while cursor < len(valid_x):
        outputs = network(Variable(valid_x[cursor:min(cursor+batch_size, len(valid_x))]), train_mode=False)
        labels = valid_y[cursor:min(cursor+batch_size, len(valid_x))]
        _, predicted = torch.max(outputs.data, 1)
        total += len(labels)
        correct += (predicted == labels).sum()
        cursor += batch_size

    print('For epoch %d \tAccuracy on valid set: %f %%' % (epoch, 100.0 * correct / total))
