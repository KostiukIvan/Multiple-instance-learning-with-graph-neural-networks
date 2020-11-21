import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, ClusterGCNConv, max_pool_neighbor_x, max_pool, dense_diff_pool, DenseSAGEConv
from torch_geometric.data import NeighborSampler, Data
import torch_geometric.nn as pyg_nn
import torch_geometric.utils as pyg_ut
from chamferdist import ChamferDistance

     

class GraphBased(nn.Module):
    def __init__(self):
        super(GraphBased, self).__init__()
        self.L = 500
        self.C = 2 # number of clusters
        self.classes = 1 # number of classes

        
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
                    
        self.gnn_pool = DenseSAGEConv(self.L, self.C)
        self.bn2 = torch.nn.BatchNorm1d(self.C)
        # self.mlp = nn.Linear(self.C, self.C, bias=True) 
        
        input_layers = int(self.L * self.C)
        hidden_layers = int(self.L * self.C / 2)
        output_layer = self.classes
        self.lin1 = nn.Linear(input_layers, hidden_layers, bias=True) 
        self.lin2 = nn.Linear(hidden_layers, output_layer, bias=True)


    def forward(self, x):
        
        turn_on_logs = False
        x = x.squeeze(0) # [9, 1, 28, 28]
        
        H = self.feature_extractor_part1(x) # [9, 50, 4, 4]
        H = H.view(-1, 50 * 4 * 4) # [9, 800]
        H = self.feature_extractor_part2(H)  # NxL  [9, 500]
        

        X, E_idx = self.convert_bag_to_graph_(H, self.n) # nodes [9, 500], E_idx [2, A]
        E_idx = E_idx.cuda()
        A = pyg_ut.to_dense_adj(E_idx, max_num_nodes=X.shape[0])

        # Embedding
        Z = F.leaky_relu(self.gnn_embd(X, A), negative_slope=0.01)
        print("Z ", Z.shape)
        Z = self.bn1(Z.squeeze())
        
        # Clustering
        S = F.leaky_relu(self.gnn_pool(X, A), negative_slope=0.01)
        print("S ", S.shape)
        S = self.bn2(S.view(-1, self.C))
        S = torch.softmax(S, dim=1)
        #S = F.leaky_relu(self.mlp(S))

        # Coarsened graph   
        adj_matrix = pyg_ut.to_dense_adj(E_idx, max_num_nodes=X.shape[0]) # Generate adjacency matrix K x K !!!!! KOSS !!!!!!
        X, adj_matrix, l1, e1 = dense_diff_pool(Z, adj_matrix, S)

        # Embedding 2
        X = F.leaky_relu(self.gnn_embd(X, adj_matrix), negative_slope=0.01) # [C, 500]
        print("X " , X.view(self.C, self.L).shape)
        X = self.bn1(X.view(self.C, self.L))
        
        # Concat
        X = X.view(1, -1)
        
        # MLP
        X = F.leaky_relu(self.lin1(X), 0.01)
        X = F.leaky_relu(self.lin2(X), 0.01)
        
        
        Y_prob = X.squeeze() # torch.max(F.softmax(X.squeeze(), dim=0))
        Y_hat = torch.ge(Y_prob, 0.5).float()
        Y_hat = torch.argmax(F.softmax(X.squeeze(), dim=0))
        if True:
            print("Y_prob : ", Y_prob)
            print("Y_hat : ", Y_hat)
        

        return Y_prob, Y_hat
    
    # GNN methods
    def convert_bag_to_graph_(self, bag, N):
        edge_index = []
        chamferDist = ChamferDistance()
        for cur_i, cur_node in enumerate(bag):
            for alt_i, alt_node in enumerate(bag):
                # if cur_i != alt_i and self.euclidean_distance_(cur_node, alt_node) < N:
                if cur_i != alt_i and chamferDist(cur_node.view(1, 1, -1), alt_node.view(1, 1, -1)) < N:
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
        _, Y_hat = self.forward(X)
        error = 1. - Y_hat.eq(Y).cpu().float().mean().data

        return error, Y_hat

    def calculate_objective(self, X, Y):
        Y = Y.float()
        Y_prob, _ = self.forward(X)
        Y_prob = torch.clamp(Y_prob, min=1e-5, max=1. - 1e-5)
        neg_log_likelihood = -1. * (Y * torch.log(Y_prob) + (1. - Y) * torch.log(1. - Y_prob))  # negative log bernoulli

        return neg_log_likelihood




class Attention(nn.Module):
    def __init__(self):
        super(Attention, self).__init__()
        self.L = 500
        self.D = 128
        self.K = 1

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

        self.attention = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Tanh(),
            nn.Linear(self.D, self.K)
        )

        self.classifier = nn.Sequential(
            nn.Linear(self.L*self.K, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = x.squeeze(0) # [9, 1, 28, 28]
        print("XXXX: ", x.shape)
        
        H = self.feature_extractor_part1(x) # [9, 50, 4, 4]
        H = H.view(-1, 50 * 4 * 4) # [9, 800]
        H = self.feature_extractor_part2(H)  # NxL  [9, 500]

        A = self.attention(H)  # NxK  [9, 1]

        A = torch.transpose(A, 1, 0)  # KxN  [1, 9]
        A = F.softmax(A, dim=1)  # softmax over N  [1, 9]

        M = torch.mm(A, H)  # KxL  [1, 500]

        Y_prob = self.classifier(M) # 0.xxxx
        Y_hat = torch.ge(Y_prob, 0.5).float() # 0 or 1
        print("Y porb ", Y_prob)
        print("Y porb ", Y_prob.shape)
        print("Y hat ", Y_hat)
        print("Y hat ", Y_hat.shape)
        return Y_prob, Y_hat, A

    # AUXILIARY METHODS
    def calculate_classification_error(self, X, Y):
        Y = Y.float()
        _, Y_hat, _ = self.forward(X)
        error = 1. - Y_hat.eq(Y).cpu().float().mean().data

        return error, Y_hat

    def calculate_objective(self, X, Y):
        Y = Y.float()
        Y_prob, _, A = self.forward(X)
        Y_prob = torch.clamp(Y_prob, min=1e-5, max=1. - 1e-5)
        neg_log_likelihood = -1. * (Y * torch.log(Y_prob) + (1. - Y) * torch.log(1. - Y_prob))  # negative log bernoulli

        return neg_log_likelihood, A

class GatedAttention(nn.Module):
    def __init__(self):
        super(GatedAttention, self).__init__()
        self.L = 500
        self.D = 128
        self.K = 1

        self.feature_extractor_part1 = nn.Sequential(
            nn.Conv2d(1, 20, kernel_size=5),            # 
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

        self.attention_V = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Tanh()
        )

        self.attention_U = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Sigmoid()
        )

        self.attention_weights = nn.Linear(self.D, self.K)

        self.classifier = nn.Sequential(
            nn.Linear(self.L*self.K, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = x.squeeze(0)

        H = self.feature_extractor_part1(x)
        H = H.view(-1, 50 * 4 * 4)
        H = self.feature_extractor_part2(H)  # NxL

        A_V = self.attention_V(H)  # NxD
        A_U = self.attention_U(H)  # NxD
        A = self.attention_weights(A_V * A_U) # element wise multiplication # NxK
        A = torch.transpose(A, 1, 0)  # KxN
        A = F.softmax(A, dim=1)  # softmax over N

        M = torch.mm(A, H)  # KxL

        Y_prob = self.classifier(M)
        Y_hat = torch.ge(Y_prob, 0.5).float()

        return Y_prob, Y_hat, A

    # AUXILIARY METHODS
    def calculate_classification_error(self, X, Y):
        Y = Y.float()
        _, Y_hat, _ = self.forward(X)
        error = 1. - Y_hat.eq(Y).cpu().float().mean().item()

        return error, Y_hat

    def calculate_objective(self, X, Y):
        Y = Y.float()
        Y_prob, _, A = self.forward(X)
        Y_prob = torch.clamp(Y_prob, min=1e-5, max=1. - 1e-5)
        neg_log_likelihood = -1. * (Y * torch.log(Y_prob) + (1. - Y) * torch.log(1. - Y_prob))  # negative log bernoulli

        return neg_log_likelihood, A
    
    
    
    
