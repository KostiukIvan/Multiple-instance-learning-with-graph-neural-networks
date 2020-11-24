import os.path as osp
from math import ceil

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data_utils
from torch_geometric.datasets import TUDataset
import torch_geometric.transforms as T
import torch_geometric.utils as pyg_ut
from torch_geometric.data import DenseDataLoader
from torch_geometric.nn import DenseSAGEConv, dense_diff_pool


from torch.autograd import Variable
from dataloader import MnistBags
from chamferdist import ChamferDistance

max_nodes = 150


class MyFilter(object):
    def __call__(self, data):
        return data.num_nodes <= max_nodes

'''
path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data',
                'PROTEINS_dense')
dataset = TUDataset(path, name='PROTEINS', transform=T.ToDense(max_nodes),
                    pre_filter=MyFilter())
dataset = dataset.shuffle()
n = (len(dataset) + 9) // 10
test_dataset = dataset[:n]
val_dataset = dataset[n:2 * n]
train_dataset = dataset[2 * n:]
test_loader = DenseDataLoader(test_dataset, batch_size=20)
val_loader = DenseDataLoader(val_dataset, batch_size=20)
train_loader = DenseDataLoader(train_dataset, batch_size=20)
'''

def load_train_test():
    print('Load Train and Test Set')
    
    train_loader = data_utils.DataLoader(MnistBags(target_number=9,
                                                   mean_bag_length=10,
                                                   var_bag_length=2,
                                                   num_bag=200,
                                                   seed=1,
                                                   train=True),
                                         batch_size=1,
                                         shuffle=True)
    
    test_loader = data_utils.DataLoader(MnistBags(target_number=9,
                                                  mean_bag_length=10,
                                                  var_bag_length=2,
                                                  num_bag=200,
                                                  seed=1,
                                                  train=False),
                                        batch_size=1,
                                        shuffle=False)
    
    val_loader = data_utils.DataLoader(MnistBags(target_number=9,
                                                  mean_bag_length=10,
                                                  var_bag_length=2,
                                                  num_bag=100,
                                                  seed=1,
                                                  train=False),
                                        batch_size=1,
                                        shuffle=False)
    return train_loader, test_loader, val_loader


class GNN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels,
                 normalize=False, add_loop=False, lin=True):
        super(GNN, self).__init__()

        self.add_loop = add_loop

        self.conv1 = DenseSAGEConv(in_channels, hidden_channels, normalize)
        self.bn1 = torch.nn.BatchNorm1d(hidden_channels)
        self.conv2 = DenseSAGEConv(hidden_channels, hidden_channels, normalize)
        self.bn2 = torch.nn.BatchNorm1d(hidden_channels)
        self.conv3 = DenseSAGEConv(hidden_channels, out_channels, normalize)
        self.bn3 = torch.nn.BatchNorm1d(out_channels)

        if lin is True:
            self.lin = torch.nn.Linear(2 * hidden_channels + out_channels,
                                       out_channels)
        else:
            self.lin = None

    def bn(self, i, x):
        batch_size, num_nodes, num_channels = x.size()

        x = x.view(-1, num_channels)
        x = getattr(self, 'bn{}'.format(i))(x)
        x = x.view(batch_size, num_nodes, num_channels)
        return x

    def forward(self, x, adj):
        # batch_size, num_nodes, in_channels = x.size()

        x0 = x
        x1 = self.bn(1, F.relu(self.conv1(x0, adj)))
        x2 = self.bn(2, F.relu(self.conv2(x1, adj)))
        x3 = self.bn(3, F.relu(self.conv3(x2, adj)))

        x = torch.cat([x1, x2, x3], dim=-1)

        if self.lin is not None:
            x = F.relu(self.lin(x))

        return x


class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.L = 50
        self.C_1 = 7
        self.C_2 = 2
        self.classes = 2 # number of classes
        
        self.n = 3.5 # 0 - no-edges; infinity - fully-conected graph
        self.n_step = 0.5 # inrement n if not enoght items
        self.num_adj_parm = 0.1 # this parameter is used to define min graph adjecment len. num_adj_parm * len(bag). 0 - disable
        
        self.feature_extractor_part1 = nn.Sequential(
            nn.Conv2d(1, 20, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(20, 50, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2)
        )

        self.feature_extractor_part2 = nn.Sequential(
            nn.Linear(50 * 4 * 4, self.L),
            nn.ReLU(),
        )


        self.gnn1_pool = GNN(50, 32, self.C_1, add_loop=True)
        self.gnn1_embed = GNN(50, 24, 2, add_loop=True, lin=False)


        self.gnn2_pool = GNN(50, 32, self.C_2)
        self.gnn2_embed = GNN(50, 24, 2, lin=False)

        self.gnn3_embed = GNN(50, 32, 32, lin=False)

        self.lin1 = torch.nn.Linear(3*32, 32)
        self.lin2 = torch.nn.Linear(32, self.classes)

    def forward(self, x):
        
        x = x.squeeze(0) # [9, 1, 28, 28]

        H = self.feature_extractor_part1(x) # [9, 50, 4, 4]
        H = H.view(-1, 50 * 4 * 4) # [9, 800]
        H = self.feature_extractor_part2(H)  # NxL  [9, 500]


        x, E_idx = self.convert_bag_to_graph_(H, self.n) # nodes [9, 500], E_idx [2, A]
        adj = pyg_ut.to_dense_adj(E_idx.cuda(), max_num_nodes=x.shape[0])
        
        s = self.gnn1_pool(x, adj)
        x = self.gnn1_embed(x, adj)
        
        x, adj, l1, e1 = dense_diff_pool(x, adj, s)
        
        s = self.gnn2_pool(x, adj)
        x = self.gnn2_embed(x, adj)

        x, adj, l2, e2 = dense_diff_pool(x, adj, s)
        
        x = self.gnn3_embed(x, adj)

        x = x.mean(dim=1)
        x = F.relu(self.lin1(x))
        x = self.lin2(x)
        return F.log_softmax(x, dim=-1), l1 + l2, e1 + e2
    
    # GNN methods
    def convert_bag_to_graph_(self, bag, N):
        edge_index = []
        chamferDist = ChamferDistance()
        for cur_i, cur_node in enumerate(bag):
            for alt_i, alt_node in enumerate(bag):
                if cur_i != alt_i and self.euclidean_distance_(cur_node, alt_node) < N:
                #if cur_i != alt_i and chamferDist(cur_node.view(1, 1, -1), alt_node.view(1, 1, -1)) < N:
                    edge_index.append(torch.tensor([cur_i, alt_i]).cuda())
                    
        if len(edge_index) < self.num_adj_parm * bag.shape[0]:
            #print(f"INFO: get number of adjecment {len(edge_index)}, min len is {self.num_adj_parm * bag.shape[0]}")
            return self.convert_bag_to_graph_(bag, N = (N + self.n_step))
        
        return bag, torch.stack(edge_index).transpose(1, 0)


    def euclidean_distance_(self, X, Y):
        return torch.sqrt(torch.dot(X, X) - 2 * torch.dot(X, Y) + torch.dot(Y, Y))



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Net().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


def train(epoch):
    model.train()
    loss_all = 0
   
    for batch_idx, (data, label) in enumerate(train_loader):
        data = data.to(device)
        bag_label = label[0]
        if torch.cuda.is_available():
            data, bag_label = data.cuda(), bag_label.cuda()
        
        data, bag_label = Variable(data), Variable(bag_label)
        optimizer.zero_grad()
        '''
        print("1 ", data.x.shape, data.x)
        print("2 ", data.adj.shape, data.adj)
        print("3 ", data.mask.shape, data.mask)
        '''
        output, _, _ = model(data)
        if bag_label:
            target = torch.tensor([1.], dtype=torch.long).cuda()
        else:
            target = torch.tensor([0.], dtype=torch.long).cuda()
        
        loss = F.nll_loss(output, target)
        loss.backward()
        loss_all += target.size(0) * loss.item()
        optimizer.step()
    return loss_all / len(train_loader)


@torch.no_grad()
def test(loader):
    model.eval()
    correct = 0

    for batch_idx, (data, label) in enumerate(loader):
        data = data.to(device)
        bag_label = label[0]
        if torch.cuda.is_available():
            data, bag_label = data.cuda(), bag_label.cuda()
            
        if bag_label:
            target = torch.tensor([1.], dtype=torch.long).cuda()
        else:
            target = torch.tensor([0.], dtype=torch.long).cuda()
            
        pred = model(data)[0].max(dim=1)[1]
        
        correct += pred.eq(target.view(-1)).sum().item()
    return correct / len(loader.dataset)



train_loader, test_loader, val_loader = load_train_test()

best_val_acc = test_acc = 0
for epoch in range(1, 151):
    train_loss = train(epoch)
    val_acc = test(val_loader)
    if val_acc > best_val_acc:
        test_acc = test(test_loader)
        best_val_acc = val_acc
    print('Epoch: {:03d}, Train Loss: {:.7f}, '
          'Val Acc: {:.7f}, Test Acc: {:.7f}'.format(epoch, train_loss,
                                                     val_acc, test_acc))