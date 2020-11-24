import os.path as osp
from math import ceil

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data_utils

import torchvision.models as models
import torchvision.transforms as transforms

from torch_geometric.datasets import TUDataset
import torch_geometric.transforms as T
import torch_geometric.utils as pyg_ut
from torch_geometric.data import DenseDataLoader
from torch_geometric.nn import DenseSAGEConv, dense_diff_pool


from torch.autograd import Variable
from dataloader import MnistBags
from chamferdist import ChamferDistance
from PIL import Image



max_nodes = 150


class MyFilter(object):
    def __call__(self, data):
        return data.num_nodes <= max_nodes
    
scaler = transforms.Scale((224, 224))
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
to_tensor = transforms.ToTensor()

def get_vector(model, layer, image):

    # 2. Create a PyTorch Variable with the transformed image
    t_img = Variable(normalize(scaler(image).repeat(3,1,1)).unsqueeze(0))

    # 3. Create a vector of zeros that will hold our feature vector
    #    The 'avgpool' layer has an output size of 512
    my_embedding = torch.zeros(512)
    # 4. Define a function that will copy the output of a layer
    def copy_data(m, i, o):
        my_embedding.copy_(o.data.reshape(o.data.size(1)))
    # 5. Attach that function to our selected layer
    h = layer.register_forward_hook(copy_data)
    # 6. Run the model on our transformed image
    model(t_img)
    # 7. Detach our copy function from the layer
    h.remove()
    # 8. Return the feature vector
    return my_embedding

def load_train_test():
    print('Load Train and Test Set')
    
    train_loader = data_utils.DataLoader(MnistBags(target_number=9,
                                                   mean_bag_length=10,
                                                   var_bag_length=2,
                                                   num_bag=300,
                                                   seed=1,
                                                   train=True),
                                         batch_size=1,
                                         shuffle=True)
    
    test_loader = data_utils.DataLoader(MnistBags(target_number=9,
                                                  mean_bag_length=10,
                                                  var_bag_length=2,
                                                  num_bag=100,
                                                  seed=1,
                                                  train=False),
                                        batch_size=1,
                                        shuffle=False)
    
    val_loader = data_utils.DataLoader(MnistBags(target_number=9,
                                                  mean_bag_length=10,
                                                  var_bag_length=2,
                                                  num_bag=50,
                                                  seed=1,
                                                  train=False),
                                        batch_size=1,
                                        shuffle=False)
    return train_loader, test_loader, val_loader


class GNN(torch.nn.Module):
    def __init__(self, in_channels, out_channels,
                 normalize=False, add_loop=False, lin=True):
        super(GNN, self).__init__()

        self.conv1 = DenseSAGEConv(in_channels, out_channels, normalize)
        self.bn1 = torch.nn.BatchNorm1d(out_channels)
 
        if lin is True:
            self.lin = torch.nn.Linear(out_channels,  out_channels)
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

        x = self.bn(1, F.relu(self.conv1(x, adj)))

        if self.lin is not None:
            x = F.relu(self.lin(x))

        return x


class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.L = 512
        self.C = 2
        self.classes = 2 # number of classes
        
        self.n = 5000000 # 0 - no-edges; infinity - fully-conected graph
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

        self.gnn1_pool = GNN(self.L, self.C, add_loop=True)
        self.gnn1_embed = GNN(self.L, self.L, add_loop=True, lin=False)


        self.gnn3_embed = GNN(self.L, self.L, lin=False)

        self.lin1 = torch.nn.Linear(self.L, 16)
        self.lin2 = torch.nn.Linear(16, self.classes)
        
        # Load the pretrained model
        self.feature_model = models.resnet18(pretrained=True).cuda()
        # Use the model object to select the desired layer
        self.feature_layer = self.feature_model._modules.get('avgpool')
        # Set model to evaluation mode
        self.feature_model.eval()   

    def forward(self, x):
        
        x = x.squeeze(0) # [9, 1, 28, 28]

        H = torch.stack([get_vector(self.feature_model, self.feature_layer, img) for img in x]).cuda()

        x, E_idx = self.convert_bag_to_graph_(H, self.n) # nodes [9, 500], E_idx [2, A]
        adj = pyg_ut.to_dense_adj(E_idx.cuda(), max_num_nodes=x.shape[0])
        
        s = self.gnn1_pool(x, adj)
        x = self.gnn1_embed(x, adj)
        
        x, adj, l1, e1 = dense_diff_pool(x, adj, s)
        
        x = self.gnn3_embed(x, adj)

        x = x.mean(dim=1)
        x = F.relu(self.lin1(x))
        x = self.lin2(x)
        return F.log_softmax(x, dim=-1), l1 , e1 
    
    # GNN methods
    def convert_bag_to_graph_(self, bag, N):
        edge_index = []
        chamferDist = ChamferDistance()
        cos = nn.CosineSimilarity(dim=1, eps=1e-6)
        for cur_i, cur_node in enumerate(bag):
            for alt_i, alt_node in enumerate(bag):
                if cur_i != alt_i and self.euclidean_distance_(cur_node, alt_node) < N:
                # if cur_i != alt_i and cos(cur_node.unsqueeze(0), alt_node.unsqueeze(0)) < N:
                # if cur_i != alt_i and chamferDist(cur_node.view(1, 1, -1), alt_node.view(1, 1, -1)) < N:
                    edge_index.append(torch.tensor([cur_i, alt_i]).cuda())
                    
        if len(edge_index) < self.num_adj_parm * bag.shape[0]:
            print(f"INFO: get number of adjecment {len(edge_index)}, min len is {self.num_adj_parm * bag.shape[0]}")
            return self.convert_bag_to_graph_(bag, N = (N + self.n_step))
        
        return bag, torch.stack(edge_index).transpose(1, 0)


    def euclidean_distance_(self, X, Y):
        return torch.sqrt(torch.dot(X, X) - 2 * torch.dot(X, Y) + torch.dot(Y, Y))



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Net().to(device)
#optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)


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
        output, l, _ = model(data)
        
        if bag_label:
            target = torch.tensor([1.], dtype=torch.long).cuda()
        else:
            target = torch.tensor([0.], dtype=torch.long).cuda()
        
        loss = F.nll_loss(output, target) + l
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
for epoch in range(1, 300):
    train_loss = train(epoch)
    val_acc = test(val_loader)
    if val_acc > best_val_acc:
        test_acc = test(test_loader)
        best_val_acc = val_acc
    print('Epoch: {:03d}, Train Loss: {:.7f}, '
          'Val Acc: {:.7f}, Test Acc: {:.7f}'.format(epoch, train_loss,
                                                     val_acc, test_acc))