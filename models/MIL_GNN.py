import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, ClusterGCNConv, global_max_pool, max_pool, dense_diff_pool, DenseSAGEConv
from torch_geometric.data import NeighborSampler, Data
import torch_geometric.nn as pyg_nn
import torch_geometric.utils as pyg_ut
from chamferdist import ChamferDistance

     
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class GraphBased28x28x1(nn.Module):
    def __init__(self):
        super(GraphBased28x28x1, self).__init__()
        self.L = 50
        self.C = 1 # number of clusters
        self.classes = 2 # number of classes

        self.n = 50 # 0 - no-edges; infinity - fully-conected graph
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


        X = H# X, E_idx = self.convert_bag_to_graph_(H, self.n) # nodes [9, 500], E_idx [2, A]
        A = torch.ones((len(X), len(X)), device=device) # A = pyg_ut.to_dense_adj(E_idx.cuda(), max_num_nodes=X.shape[0])

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
        for cur_i, cur_node in enumerate(bag):
            for alt_i, alt_node in enumerate(bag):
                if cur_i != alt_i and self.euclidean_distance_(cur_node, alt_node) < N:
                #if cur_i != alt_i and chamferDist(cur_node.view(1, 1, -1), alt_node.view(1, 1, -1)) < N:
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

    # AUXILIARY METHODS
    def calculate_classification_error(self, output, target, TP, TN, FP, FN):
        pred = torch.argmax(output)
        #error = 1. - pred.eq(target).cpu().float().mean().data
        if pred.eq(1) and target.eq(1):
            TP[0] += 1

        elif pred.eq(1) and target.eq(0):
            FP[0] += 1

        elif pred.eq(0) and target.eq(1):
            FN[0] += 1

        elif pred.eq(0) and target.eq(0):
            TN[0] += 1

    def cross_entropy_loss(self, output, target):
        output = output.unsqueeze(0) if output.dim() == 1 else output
        target = torch.tensor(target, dtype=torch.long).cuda()
        loss = nn.CrossEntropyLoss()
        l1 = loss(output, target) 
    
        return l1 
    
    def calculate_objective(self, X, target):
        target = target.float()
        Y_prob, l1 = self.forward(X)
        Y_prob = torch.clamp(Y_prob, min=1e-5, max=1. - 1e-5)
        neg_log_likelihood = -1. * (target * torch.log(Y_prob) + (1. - target) * torch.log(1. - Y_prob))  # negative log bernoulli

        return neg_log_likelihood.data[0]


class GraphBased27x27x3(nn.Module):
    def __init__(self):
        super(GraphBased27x27x3, self).__init__()
        self.L = 50
        self.C = 1 # number of clusters
        self.classes = 2 # number of classes

        
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


    def forward(self, x):
        x = x.squeeze(0) # [9, 1, 28, 28]
        x = x.unsqueeze(0) if x.dim() == 3 else x
        
        H = self.feature_extractor_part1(x) # [9, 50, 4, 4]
        H = H.view(-1, 50 * 3 * 3) # [9, 800]
        H = self.feature_extractor_part2(H)  # NxL  [9, 500]

        X = H #, E_idx = self.convert_bag_to_graph_(H, self.n) # nodes [9, 500], E_idx [2, A]
        A = torch.ones((len(X), len(X)), device=device) #pyg_ut.to_dense_adj(E_idx.cuda(), max_num_nodes=x.shape[0])

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
        for cur_i, cur_node in enumerate(bag):
            for alt_i, alt_node in enumerate(bag):
                if cur_i != alt_i and self.euclidean_distance_(cur_node, alt_node) < N:
                #if cur_i != alt_i and chamferDist(cur_node.view(1, 1, -1), alt_node.view(1, 1, -1)) < N:
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

    # AUXILIARY METHODS
    def calculate_classification_error(self, output, target, TP, TN, FP, FN):
        pred = torch.argmax(output)
        #error = 1. - pred.eq(target).cpu().float().mean().data
        if pred.eq(1) and target.eq(1):
            TP[0] += 1

        elif pred.eq(1) and target.eq(0):
            FP[0] += 1

        elif pred.eq(0) and target.eq(1):
            FN[0] += 1

        elif pred.eq(0) and target.eq(0):
            TN[0] += 1

    def cross_entropy_loss(self, output, target):
        output = output.unsqueeze(0) if output.dim() == 1 else output
        target = target.squeeze()
        target = torch.tensor([target], dtype=torch.long).cuda()

        criterion = nn.CrossEntropyLoss()
        loss = criterion(output, target)
        return loss
    
    def calculate_objective(self, X, target):
        target = target.float()
        Y_prob, l1 = self.forward(X)
        Y_prob = torch.clamp(Y_prob, min=1e-5, max=1. - 1e-5)
        neg_log_likelihood = -1. * (target * torch.log(Y_prob) + (1. - target) * torch.log(1. - Y_prob))  # negative log bernoulli

        return neg_log_likelihood.data[0]


class GraphBased50x50x3(nn.Module):
    def __init__(self):
        super(GraphBased50x50x3, self).__init__()
        self.L = 50
        self.C = 1 # number of clusters
        self.classes = 2 # number of classes

        
        self.n = 0.012 # 0 - no-edges; infinity - fully-conected graph
        self.n_step = 0.001 # inrement n if not enoght items
        self.num_adj_parm = 0.1 # this parameter is used to define min graph adjecment len. num_adj_parm * len(bag). 0 - disable

        self.feature_extractor_part1 = nn.Sequential(
            nn.Conv2d(3, 10, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(10, 20, kernel_size=5),
            nn.ReLU(),
            nn.Conv2d(20, 30, kernel_size=5),
            nn.ReLU(),
            nn.Conv2d(30, 40, kernel_size=5),
            nn.ReLU(),
            nn.Conv2d(40, 50, kernel_size=5),
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


    def forward(self, x):
        x = x.squeeze(0) # [9, 1, 28, 28]
        x = x.unsqueeze(0) if x.dim() == 3 else x
        
        H = self.feature_extractor_part1(x) # [9, 50, 4, 4]
        
        H = H.view(-1, 50 * 3 * 3) # [9, 800]
        H = self.feature_extractor_part2(H)  # NxL  [9, 500]

        X = H #, E_idx = self.convert_bag_to_graph_(H, self.n) # nodes [9, 500], E_idx [2, A] 
        A = torch.ones((len(X), len(X)), device=device) # pyg_ut.to_dense_adj(E_idx.cuda(), max_num_nodes=x.shape[0]) #
        
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
        for cur_i, cur_node in enumerate(bag):
            for alt_i, alt_node in enumerate(bag):
                if cur_i != alt_i and self.euclidean_distance_(cur_node, alt_node) < N:
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

    # AUXILIARY METHODS
    def calculate_classification_error(self, output, target, TP, TN, FP, FN):
        pred = torch.argmax(output)
        #error = 1. - pred.eq(target).cpu().float().mean().data
        if pred.eq(1) and target.eq(1):
            TP[0] += 1

        elif pred.eq(1) and target.eq(0):
            FP[0] += 1

        elif pred.eq(0) and target.eq(1):
            FN[0] += 1

        elif pred.eq(0) and target.eq(0):
            TN[0] += 1

    def cross_entropy_loss(self, output, target):
        output = output.unsqueeze(0) if output.dim() == 1 else output
        target = target.squeeze()
        target = torch.tensor([target], dtype=torch.long).cuda()

        criterion = nn.CrossEntropyLoss()
        loss = criterion(output, target)
        return loss
    
    def calculate_objective(self, X, target):
        target = target.float()
        Y_prob, l1 = self.forward(X)
        Y_prob = torch.clamp(Y_prob, min=1e-5, max=1. - 1e-5)
        neg_log_likelihood = -1. * (target * torch.log(Y_prob) + (1. - target) * torch.log(1. - Y_prob))  # negative log bernoulli

        return neg_log_likelihood.data[0]

