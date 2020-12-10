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
    
    step = N * 1 // 20
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
        self.C = 1 # number of clusters
        self.classes = 4 # number of classes

        
        self.n = 50 # 0 - no-edges; infinity - fully-conected graph
        self.n_step = 0.5 # inrement n if not enoght items
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
            nn.Linear(50 * 3 * 3, self.L),
            nn.ReLU(),
        )

        self.gnn_embd = DenseSAGEConv(self.L, self.L)    # correct : https://arxiv.org/abs/1706.02216
        self.bn1 = nn.BatchNorm1d(self.L)
                    
        self.gnn_pool = DenseSAGEConv(self.L, self.C)
        self.bn2 = torch.nn.BatchNorm1d(self.C)
        self.mlp = nn.Linear(self.C, self.C, bias=True) 
        
        self.gnn_embd2 = DenseSAGEConv(self.L, self.L)    # correct : https://arxiv.org/abs/1706.02216
        self.bn3 = nn.BatchNorm1d(self.L)
        
        input_layers = int(self.L * self.C)
        hidden_layers = int(self.L * self.C / 2)
        output_layer = self.classes
        self.lin1 = nn.Linear(input_layers, hidden_layers, bias=True) 
        self.lin2 = nn.Linear(hidden_layers, output_layer, bias=True)
        

        # Load the pretrained model
        self.feature_model = models.resnet18(pretrained=True)
        # Use the model object to select the desired layer
        self.feature_layer = self.feature_model._modules.get('avgpool')
        # Set model to evaluation mode
        self.feature_model.eval()   

    def forward(self, x):
        x = x.squeeze(0) # [9, 1, 28, 28]
        
        # H = torch.stack([get_vector(self.feature_model, self.feature_layer, img) for img in x]).cuda()
        H = self.feature_extractor_part1(x) # [9, 50, 4, 4]
        H = H.view(-1, 50 * 3 * 3) # [9, 800]
        H = self.feature_extractor_part2(H)  # NxL  [9, 500]


        X, E_idx = self.convert_bag_to_graph_(H, self.n) # nodes [9, 500], E_idx [2, A]
        A = pyg_ut.to_dense_adj(E_idx.cuda(), max_num_nodes=x.shape[0])

        # Embedding
        Z = F.leaky_relu(self.gnn_embd(X, A), negative_slope=0.01)
        loss_emb_1 = self.auxiliary_loss(A, Z)
        
        # Clustering
        S = F.leaky_relu(self.gnn_pool(X, A), negative_slope=0.01)
        S = F.leaky_relu(self.mlp(S), negative_slope=0.01)

        # Coarsened graph   
        X, A, l1, e1 = dense_diff_pool(Z, A, S)

        # Embedding 2
        X = F.leaky_relu(self.gnn_embd(X, A), negative_slope=0.01) # [C, 500]
        loss_emb_2 = self.auxiliary_loss(A, X)
        
        # Concat
        X = X.view(1, -1)

        # MLP
        X = F.leaky_relu(self.lin1(X), 0.01)
        X = F.leaky_relu(self.lin2(X), 0.01)
         
        Y_prob = F.softmax(X.squeeze(), dim=0)

        return Y_prob, (l1 + loss_emb_1 + loss_emb_2)
    
    # GNN methods
    def convert_bag_to_graph_(self, bag, N):
        edge_index = []
        chamferDist = ChamferDistance()
        cos = nn.CosineSimilarity(dim=1, eps=1e-6)
        for cur_i, cur_node in enumerate(bag):
            for alt_i, alt_node in enumerate(bag):
                # print(cos(cur_node.unsqueeze(0), alt_node.unsqueeze(0)))
                if cur_i != alt_i and self.euclidean_distance_(cur_node, alt_node) < N:
                # if cur_i != alt_i and cos(cur_node.unsqueeze(0), alt_node.unsqueeze(0)) > N:
                # if cur_i != alt_i and chamferDist(cur_node.view(1, 1, -1), alt_node.view(1, 1, -1)) < N:
                    edge_index.append(torch.tensor([cur_i, alt_i]).cuda())
                    
        if len(edge_index) < self.num_adj_parm * bag.shape[0]:
            print(f"INFO: get number of adjecment {len(edge_index)}, min len is {self.num_adj_parm * bag.shape[0]}")
            return self.convert_bag_to_graph_(bag, N = (N + self.n_step))
        
        return bag, torch.stack(edge_index).transpose(1, 0)


    def euclidean_distance_(self, X, Y):
        return torch.sqrt(torch.dot(X, X) - 2 * torch.dot(X, Y) + torch.dot(Y, Y))
    
    def auxiliary_loss(self, A, S):
        '''
            A: adjecment matrix {0,1} K x K
            S: nodes R K x D
        '''
        A = A.unsqueeze(0) if A.dim() == 2 else A
        S = S.unsqueeze(0) if S.dim() == 2 else S
        
        S = torch.softmax(S, dim=-1)
    
        link_loss = A - torch.matmul(S, S.transpose(1, 2))
        link_loss = torch.norm(link_loss, p=2)
        link_loss = link_loss / A.numel()
        
        return link_loss

    def cross_entropy_loss(self, X, target):
        Y_prob, l2 = self.forward(X)
        
        Y_prob = Y_prob.unsqueeze(0) if Y_prob.dim() == 1 else Y_prob
        target = torch.tensor(target, dtype=torch.long)
        
        print(Y_prob.shape, Y_prob)
        print(target.shape, target)
        loss = nn.CrossEntropyLoss()
        l1 = loss(Y_prob, target) 
    
        return l1 + l2

    def negative_log_likelihood_loss(self, X, target):
        Y_prob, l2 = self.forward(X)
        
        Y_prob = Y_prob.unsqueeze(0) if Y_prob.dim() == 1 else Y_prob
        target = torch.tensor(target, dtype=torch.long)
        
        loss = nn.NLLLoss()
        l1 = -1 * loss(Y_prob, target) 
    
        return l1 + l2
    
    def calculate_objective(self, X, target):
        target = target.float()
        Y_prob, l1 = self.forward(X)
        Y_prob = torch.clamp(Y_prob, min=1e-5, max=1. - 1e-5)
        neg_log_likelihood = -1. * (target * torch.log(Y_prob) + (1. - target) * torch.log(1. - Y_prob))  # negative log bernoulli

        return neg_log_likelihood.data[0]



model = Net().cuda()

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.BCELoss()

train_loader, test_loader, val_loader = load_train_test_val(ds) 

def train(train_loader):
    model.train()
    loss_all = 0
    correct = 0
    
    for batch_idx, (data, target) in enumerate(train_loader):
        print("|", end="")
        target = torch.tensor(target)
        if torch.cuda.is_available():
            data, target = data.cuda(), target.cuda()
        optimizer.zero_grad()
        output, l = model(data)
        loss = criterion(output.squeeze(), target.type(torch.float)) + l
        #loss = model.cross_entropy_loss(data, target)
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

        pred, _ = model(data)  
        pred = torch.ge(pred, 0.5)
        correct += pred.eq(target.view(-1)).sum().item()
   
    print()
    return correct / (len(loader) * len(loader[0][1]))




best_val_acc = test_acc = 0
for epoch in range(1, 300):
    train_loss, train_acc = train(train_loader)
    val_acc = test(val_loader)
    if val_acc > best_val_acc:
        test_acc = test(test_loader)
        best_val_acc = val_acc
    print('Epoch: {:03d}, Train Loss: {:.7f}, Train acc: {:.7f}, '
          'Val Acc: {:.7f}, Test Acc: {:.7f}'.format(epoch, train_loss, train_acc,
                                                     val_acc, test_acc))