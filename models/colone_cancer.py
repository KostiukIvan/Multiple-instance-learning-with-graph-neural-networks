import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data_utils
import torchvision.transforms as transforms
import torchvision.models as models
import torch_geometric.utils as pyg_ut
from torch_geometric.nn import DenseSAGEConv, dense_diff_pool

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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

    def forward(self, x, adj, mask=None):
        x0 = x
        x1 = self.bn(1, F.relu(self.conv1(x0, adj, mask)))
        x2 = self.bn(2, F.relu(self.conv2(x1, adj, mask)))
        x3 = self.bn(3, F.relu(self.conv3(x2, adj, mask)))

        x = torch.cat([x1, x2, x3], dim=-1)

        if self.lin is not None:
            x = F.relu(self.lin(x))

        return x

edge_pairs_dynamic = {}
class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.L = 50
        self.C = 4 # number of clusters
        self.classes = 2 # number of classes

        self.n = 50 # 0 - no-edges; infinity - fully-conected graph
        self.n_step = 0.5 # inrement n if not enoght items
        self.num_adj_parm = 1 # this parameter is used to define min graph adjecment len. num_adj_parm * len(bag). 0 - disable

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

        self.gnn_pool = GNN(self.L, 64, self.C, add_loop=True)
        self.gnn_embed = GNN(self.L, 64, self.L, add_loop=True, lin=False)

        self.gnn_embed_2 = GNN(2 * 64 + self.L, 64, self.L, add_loop=True, lin=False)

        input_layers = int((2 * 64 + self.L) * self.C)
        self.lin1 = nn.Linear(input_layers, int(input_layers / 2), bias=True) 
        self.lin2 = nn.Linear(int(input_layers / 2), self.classes, bias=True)

    def forward(self, x):
        x = x.squeeze(0) # [9, 1, 28, 28]
        x = x.unsqueeze(0) if x.dim() == 3 else x
        
        H = self.feature_extractor_part1(x) # [9, 50, 4, 4]
        H = H.view(-1, 50 * 3 * 3) # [9, 800]
        H = self.feature_extractor_part2(H)  # NxL  [9, 500]

        X = H #, E_idx = self.convert_bag_to_graph_(H, self.n) # nodes [9, 500], E_idx [2, A]
        A = torch.ones((len(X), len(X)), device=device) #pyg_ut.to_dense_adj(E_idx.cuda(), max_num_nodes=x.shape[0])

        # Embedding
        Z = F.leaky_relu(self.gnn_embed(X, A), negative_slope=0.01)
        loss_emb_1 = self.auxiliary_loss(A, Z)
        
        # Clustering
        S = F.leaky_relu(self.gnn_pool(X, A), negative_slope=0.01)

        # Coarsened graph   
        X, A, l1, e1 = dense_diff_pool(Z, A, S)

        # Embedding 2
        X = F.leaky_relu(self.gnn_embed_2(X, A), negative_slope=0.01) # [C, 500]
        loss_emb_2 = self.auxiliary_loss(A, X)

        # Concat
        X = X.view(1, -1)

        # MLP
        X = F.leaky_relu(self.lin1(X), 0.01)
        X = F.leaky_relu(self.lin2(X), 0.01)
         
        Y_prob = X 
        return Y_prob, (l1 + loss_emb_1 + loss_emb_2)
    
    # GNN methods
    def convert_bag_to_graph_(self, bag, N):
        l = len(bag)
        
        if l in edge_pairs_dynamic:
            edge_index = edge_pairs_dynamic[l]
        else:
            edge_index = [torch.tensor([cur_i, alt_i], device=device) for cur_i, alt_i in itertools.product(range(len(bag)), repeat=2)]
            edge_pairs_dynamic[l] = edge_index

         
        if len(edge_index) < self.num_adj_parm * bag.shape[0]:
            print(f"INFO: get number of adjecment {len(edge_index)}, min len is {self.num_adj_parm * bag.shape[0]}")
            return self.convert_bag_to_graph_(bag, N = (N + self.n_step))
        return bag, torch.stack(edge_index).transpose(1, 0)

    
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

    def cross_entropy_loss(self, output, target):  
        output = output.unsqueeze(0) if output.dim() == 1 else output
        target = target.squeeze()
        target = torch.tensor([target], dtype=torch.long).cuda()
        
        criterion = nn.CrossEntropyLoss()
        loss = criterion(output, target)
        return loss

    def MSE(self, output, target):
        output = output.unsqueeze(0) if output.dim() == 1 else output
        target = target.unsqueeze(0) if target.dim() == 1 else target
 
        criterion = nn.MSELoss()

        loss = criterion(output, target)

        return loss

    def L1(self, output, target):
        output = output.unsqueeze(0) if output.dim() == 1 else output
        target = target.unsqueeze(0) if target.dim() == 1 else target
 
        criterion = nn.L1Loss()
        loss = criterion(output, target)

        return loss


    def negative_log_likelihood_loss(self, output, target):
        output = output.unsqueeze(0) if output.dim() == 1 else output
        target = target.unsqueeze(0) if target.dim() == 1 else target
        
        loss = nn.NLLLoss()
        l1 = -1 * loss(output, target) 
    
        return l1 
    
    def calculate_objective(self, X, target):
        target = target.float()
        Y_prob, l1 = self.forward(X)
        Y_prob = torch.clamp(Y_prob, min=1e-5, max=1. - 1e-5)
        neg_log_likelihood = -1. * (target * torch.log(Y_prob) + (1. - target) * torch.log(1. - Y_prob))  # negative log bernoulli

        return neg_log_likelihood.data[0]

    def calculate_classification_error(self, output, target):
        pred = torch.argmax(output)
        error = 1. - pred.eq(target).cpu().float().mean().data

        return error