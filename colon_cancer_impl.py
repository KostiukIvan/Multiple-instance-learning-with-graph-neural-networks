import os
import random

import scipy.io
import numpy as np 

from PIL import Image
from skimage import io, color
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data_utils

import torchvision.transforms as transforms
import torchvision.models as models


import torch_geometric.utils as pyg_ut
from torch_geometric.nn import DenseSAGEConv, dense_diff_pool

import python_data.utils_augmentation as utils_augmentation


from torch.autograd import Variable
from dataloader import MnistBags
from chamferdist import ChamferDistance
from colon_dataset import ColonCancerBagsCross

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

ds = ColonCancerBagsCross(path='C:\\Users\\ivank\\UJ\\Computer Vision\\Final Project\\MIL_wiht_GNN\\python_data\\ColonCancer\\', train_val_idxs=range(100), test_idxs=[], loc_info=False)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

scaler = transforms.Resize((224, 224))
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
to_tensor = transforms.ToTensor()


def load_train_test_val(ds):
    N = len(ds)
    train = []
    test = []
    val = []
    
    step = N * 1 // 2
    [train.append((ds[i][0], ds[i][1][0])) for i in range(0, step)]
    print(f"train loaded {len(train)} items")
   
    [val.append((ds[i][0], ds[i][1][0])) for i in range(step, step + step // 5)]
    print(f"valid loaded {len(val)} items")

    [test.append((ds[i][0], ds[i][1][0])) for i in range(step,  step + step // 2)]
    print(f"test loaded {len(test)} items")
    
    return train, test, val



def get_vector(model, layer, image):

    # 2. Create a PyTorch Variable with the transformed image
    t_img = Variable(normalize(scaler(image)).unsqueeze(0))

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

        x = self.bn(1, F.leaky_relu(self.conv1(x, adj), negative_slope=0.01))

        if self.lin is not None:
            x = F.leaky_relu(self.lin(x), negative_slope=0.01)

        return x


class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.L = 512
        self.C = 2
        self.CS = 4 # number of classes
        
        self.n = 0.5 # 0 - no-edges; infinity - fully-conected graph
        self.n_step = 0.01 # inrement n if not enoght items
        self.num_adj_parm = 0.1 # this parameter is used to define min graph adjecment len. num_adj_parm * len(bag). 0 - disable
        
        self.feature_extractor_part1 = nn.Sequential(
            nn.Conv2d(3, 20, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(20, 50, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2)
        )

        self.feature_extractor_part2 = nn.Sequential(
            nn.Linear(450, self.L),
            nn.ReLU(),
        )

        self.gnn1_pool = GNN(self.L, self.C)
        self.gnn1_embed = GNN(self.L, self.L, lin=False)


        self.gnn3_embed = GNN(self.L, self.L, lin=False)

        input_layers = int(self.L)
        hidden_layers = int(self.L / 2)
        self.lin1 = torch.nn.Linear(input_layers, hidden_layers)
        self.lin2 = torch.nn.Linear(hidden_layers, self.CS)
        
        # Load the pretrained model
        self.feature_model = models.resnet18(pretrained=True)
        # Use the model object to select the desired layer
        self.feature_layer = self.feature_model._modules.get('avgpool')
        # Set model to evaluation mode
        self.feature_model.eval()   

    def forward(self, x):
        x = x.squeeze(0) # [9, 1, 28, 28]
        
        H = torch.stack([get_vector(self.feature_model, self.feature_layer, img) for img in x]).cuda()
        # dim = x.shape[0]
        # H = self.feature_extractor_part1(x) # [9, 50, 4, 4]
        # H = H.view(dim,-1) # [9, 800]
        # H = self.feature_extractor_part2(H)  # NxL  [9, 500]
        # H = x.view(-1, 28 * 28)

        x, E_idx = self.convert_bag_to_graph_(H, self.n) # nodes [9, 500], E_idx [2, A]
        adj = pyg_ut.to_dense_adj(E_idx.cuda(), max_num_nodes=x.shape[0])

        s = self.gnn1_pool(x, adj)
        x = self.gnn1_embed(x, adj)
        
        x, adj, l1, e1 = dense_diff_pool(x, adj, s)
        
        x = self.gnn3_embed(x, adj)

        x = x.mean(dim=1)
        x = F.relu(self.lin1(x))
        x = self.lin2(x)
        
        return F.softmax(x, dim=-1), l1 , e1 
    
    # GNN methods
    def convert_bag_to_graph_(self, bag, N):
        edge_index = []
        chamferDist = ChamferDistance()
        cos = nn.CosineSimilarity(dim=1, eps=1e-6)
        for cur_i, cur_node in enumerate(bag):
            for alt_i, alt_node in enumerate(bag):
                # print(cos(cur_node.unsqueeze(0), alt_node.unsqueeze(0)))
                # if cur_i != alt_i and self.euclidean_distance_(cur_node, alt_node) < N:
                if cur_i != alt_i and cos(cur_node.unsqueeze(0), alt_node.unsqueeze(0)) > N:
                # if cur_i != alt_i and chamferDist(cur_node.view(1, 1, -1), alt_node.view(1, 1, -1)) < N:
                    edge_index.append(torch.tensor([cur_i, alt_i]).cuda())
                    
        if len(edge_index) < self.num_adj_parm * bag.shape[0]:
            print(f"INFO: get number of adjecment {len(edge_index)}, min len is {self.num_adj_parm * bag.shape[0]}")
            return self.convert_bag_to_graph_(bag, N = (N - self.n_step))
        
        return bag, torch.stack(edge_index).transpose(1, 0)


    def euclidean_distance_(self, X, Y):
        return torch.sqrt(torch.dot(X, X) - 2 * torch.dot(X, Y) + torch.dot(Y, Y))


model = Net().cuda()

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.BCELoss()

train_loader, test_loader, val_loader = load_train_test_val(ds) 

def train(epoch):
    model.train()
    loss_all = 0
    correct = 0
    
    for batch_idx, (data, target) in enumerate(train_loader):
        print("|", end="")
        target = torch.tensor(target)
        if torch.cuda.is_available():
            data, target = data.cuda(), target.cuda()
        optimizer.zero_grad()
        output, l, _ = model(data)
        loss = criterion(output.squeeze(), target.type(torch.float))  + l
        loss.backward()
        loss_all += target.size(0) * loss.item()
        optimizer.step()
    print()
    return loss_all / len(train_loader), 0


@torch.no_grad()
def test(loader):
    model.eval()
    correct = 0
    for batch_idx, (data, target) in enumerate(loader):
        print("|", end="")
        target = torch.tensor(target)
        if torch.cuda.is_available():
            data, target = data.cuda(), target.cuda()

        pred, _, _ = model(data)  
        correct += pred.eq(target.view(-1)).sum().item()
        
    print()
    return correct / (len(loader) * len(loader[0][1]))




best_val_acc = test_acc = 0
for epoch in range(1, 300):
    train_loss, train_acc = train(epoch)
    val_acc = test(val_loader)
    if val_acc > best_val_acc:
        test_acc = test(test_loader)
        best_val_acc = val_acc
    print('Epoch: {:03d}, Train Loss: {:.7f}, Train acc: {:.7f}, '
          'Val Acc: {:.7f}, Test Acc: {:.7f}'.format(epoch, train_loss, train_acc,
                                                     val_acc, test_acc))