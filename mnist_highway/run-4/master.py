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
        super(Highway, self).__init__()
        # Affine transformation layer and transform gates
        self.linear = nn.Linear(fan_in, fan_out)
        self.transform = nn.Linear(fan_in, fan_out)
        self.batch_norm = nn.BatchNorm1d(fan_out)
        # Get weight initialization function
        w_initialization = getattr(nn.init, w_init)
        w_initialization(self.linear.weight)
        w_initialization(self.transform.weight)
        nn.init.constant(self.transform.bias, b_init)
        nn.init.uniform(self.linear.bias)

    def forward(self, x, train_mode=True):
        h = F.leaky_relu(self.linear(x))
        t = F.sigmoid(self.transform(x))
        self.batch_norm.training = train_mode
        return self.batch_norm(h * t + (1 - t) * x), t


class Net(nn.Module):
    def __init__(self, fan_in=784, fan_out=150):
        super(Net, self).__init__()
        self.linear = nn.Linear(fan_in, fan_out)
        self.highway_layers = []
        self.final = nn.Linear(fan_out, 10)
        for i in xrange(10):
            self.highway_layers.append(Highway(fan_out, fan_out).cuda())

    def forward(self, x, train_mode=True, get_t=False):
        net = F.leaky_relu(self.linear(x))
        temp = None
        t_cost = 0
        for layer in self.highway_layers:
            net,t = layer(net, train_mode)
            t_cost = torch.sum(torch.max(t, dim=0)[0])
            if get_t:
                if temp is None:
                    temp = np.expand_dims(t.numpy(), axis=1)
                else:
                    temp = np.append(temp, np.expand_dims(t.numpy(), axis=1), axis=1)
        net = self.final(net)
        if get_t:
            return net, temp.numpy()
        if train_mode:
            return net, t_cost
        return net

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
