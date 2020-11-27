import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, ClusterGCNConv, global_max_pool, max_pool, dense_diff_pool, DenseSAGEConv
from torch_geometric.data import NeighborSampler, Data
import torch_geometric.nn as pyg_nn
import torch_geometric.utils as pyg_ut
from chamferdist import ChamferDistance

     

class GraphBased(nn.Module):
    def __init__(self):
        super(GraphBased, self).__init__()
        self.L = 50
        self.C = 1 # number of clusters
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

        self.gnn_embd = DenseSAGEConv(self.L, self.L)    # correct : https://arxiv.org/abs/1706.02216
        self.bn1 = nn.BatchNorm1d(self.L)
        #self.pool1 = nn.MaxPool1d()(3, stride=2)
                    
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


    def forward(self, x):
        x = x.squeeze(0) # [9, 1, 28, 28]
        H = self.feature_extractor_part1(x) # [9, 50, 4, 4]
        H = H.view(-1, 50 * 4 * 4) # [9, 800]
        H = self.feature_extractor_part2(H)  # NxL  [9, 500]


        X, E_idx = self.convert_bag_to_graph_(H, self.n) # nodes [9, 500], E_idx [2, A]
        A = pyg_ut.to_dense_adj(E_idx.cuda(), max_num_nodes=X.shape[0])

        # Embedding
        Z = F.leaky_relu(self.gnn_embd(X, A), negative_slope=0.01)

        # Clustering
        S = F.leaky_relu(self.gnn_pool(X, A), negative_slope=0.01)
        S = F.leaky_relu(self.mlp(S), negative_slope=0.01)

        # Coarsened graph   
        X, adj_matrix, l1, e1 = dense_diff_pool(Z, A, S)

        # Embedding 2
        X = F.leaky_relu(self.gnn_embd(X, adj_matrix), negative_slope=0.01) # [C, 500]
        # loss_emb_2 = self.pool1(X)
        
        # Concat
        X = X.view(1, -1)

        # MLP
        X = F.leaky_relu(self.lin1(X), 0.01)
        X = F.leaky_relu(self.lin2(X), 0.01)
        
       
        Y_prob = torch.max(F.softmax(X.squeeze(), dim=0))
        #Y_hat = torch.ge(F.softmax(X.squeeze(), dim=0), 0.5).float()
        Y_hat = torch.argmax(F.softmax(X.squeeze(), dim=0))
        if False:
            print("Y_prob : ", Y_prob)
            print("Y_hat : ", Y_hat)
        

        return Y_prob, Y_hat, l1 
    
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

    # AUXILIARY METHODS
    def calculate_classification_error(self, X, Y):
        Y = Y.float()
        _, Y_hat, _ = self.forward(X)
        error = 1. - Y_hat.eq(Y).cpu().float().mean().data

        return error, Y_hat

    def calculate_objective(self, X, Y):
        Y = Y.float()
        Y_prob, _, l1 = self.forward(X)
        Y_prob = torch.clamp(Y_prob, min=1e-5, max=1. - 1e-5)
        neg_log_likelihood = -1. * (Y * torch.log(Y_prob) + (1. - Y) * torch.log(1. - Y_prob))  # negative log bernoulli

        return neg_log_likelihood 




