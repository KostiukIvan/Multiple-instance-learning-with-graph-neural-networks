import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, ClusterGCNConv, max_pool_neighbor_x, max_pool
from torch_geometric.data import NeighborSampler, Data
import torch_geometric.nn as pyg_nn
import torch_geometric.utils as pyg_ut

class GraphBased(nn.Module):
    def __init__(self):
        super(GraphBased, self).__init__()
        self.L = 500
        self.D = 128
        self.K = 1
        
        self.n = 2.5 # 0 - no-edges; infinity - fully-conected graph
        self.n_step = 0.5 # inrement n if not enoght items
        self.num_adj_parm = 0.1 # this parameter is used to define min graph adjecment len. num_adj_parm * len(bag). 0 - disable
        self.C = 5 # number of clusters
        self.classes = 2 # number of classes

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

        self.gnn_embd = SAGEConv(500, 500)
        self.gnn_embd_bn = nn.BatchNorm1d(500)
        
        self.gnn_clust = ClusterGCNConv(500, self.C)
        
        self.lin1 = nn.Linear(500, 250, bias=True) 
        self.lin2 = nn.Linear(250, 1, bias=True)

        self.classifier = nn.Sequential(
            nn.Linear(self.L*self.K, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        
        turn_on_logs = False
        x = x.squeeze(0) # [9, 1, 28, 28]
        
        H = self.feature_extractor_part1(x) # [9, 50, 4, 4]
        H = H.view(-1, 50 * 4 * 4) # [9, 800]
        H = self.feature_extractor_part2(H)  # NxL  [9, 500]
        if turn_on_logs:
            print("H = self.feature_extractor_part2(H): ", H.shape)
            print(H)
            print("============================================================================================")

        nodes, edge_index = self.convert_bag_to_graph_(H, self.n) # nodes [9, 500], edge_index [2, A]
        edge_index = edge_index.cuda()
        if turn_on_logs:
            print("Nodes", nodes.shape, " Edges", edge_index.shape)
            print("Nodes : ", nodes)
            print("Edges : ", edge_index)
            print("------------------------------------------------------------------------------")

        # Embedding
        Z = self.gnn_embd(nodes, edge_index)
        Z = F.leaky_relu(Z, negative_slope=0.01)
        Z = self.gnn_embd_bn(Z)
        if turn_on_logs:
            print("Embeding : ", Z.shape)
            print("Value : ", Z)
            print("------------------------------------------------------------------------------")
        
        # Clustering
        S = self.gnn_clust(nodes, edge_index)
        S = S.softmax(dim=1)
        if turn_on_logs:
            print("Clustering : ", S.shape)
            print("Value : ", S)
            print("------------------------------------------------------------------------------")
        
        # Coarsened graph
        coar_nodes = S.transpose(0,1) @ Z # [C, 500]
        A = pyg_ut.to_dense_adj(edge_index, max_num_nodes=nodes.shape[0]) # Generate adjacency matrix K x K !!!!! KOSS !!!!!!
        coar_edges = S.transpose(0,1) @ A @ S
        
        coar_edges = coar_edges.squeeze()
        coar_edges[torch.where(coar_edges < torch.mean(coar_edges))] = 0.0
        coar_edges, coar_edges_val = pyg_ut.dense_to_sparse(coar_edges.squeeze()) # [2, 2500] could be modified, look to conversation on teams
        if turn_on_logs:
            print("Coarsened graph : ", coar_nodes.shape)
            print("Value : ", coar_nodes)
            print("Edges : ", coar_edges.shape, coar_edges)
            print("------------------------------------------------------------------------------")
        
        # Embedding 2
        embd_nodes = self.gnn_embd(coar_nodes, coar_edges) 
        embd_nodes = F.leaky_relu(embd_nodes, negative_slope=0.01)
        embd_nodes = self.gnn_embd_bn(embd_nodes)# [C, 500]
        if turn_on_logs:
            print("Embeding 2 : ", embd_nodes.shape, embd_nodes)
            print("------------------------------------------------------------------------------")
        
        # Max pool
        graph = Data(embd_nodes, coar_edges) # graph creation
        graph = max_pool_neighbor_x(graph) # !!!!! KOSS !!!!!! 
        if turn_on_logs:
            print("Max pool : ", graph.x.shape, graph.x)
            print("------------------------------------------------------------------------------")
        
        # MLP
        x = graph.x
        x = F.leaky_relu(self.lin1(x), 0.01)
        x = F.leaky_relu(self.lin2(x), 0.01)
        if turn_on_logs:
            print("MLP : ", x.shape, x)
            print("------------------------------------------------------------------------------")
        
        
        Y_prob = torch.max(F.softmax(x.squeeze(), dim=0))
        Y_hat = torch.ge(Y_prob, 0.5).float()
        if turn_on_logs:
            print("Y_prob : ", Y_prob)
            print("Y_hat : ", Y_hat)
            
        '''
        Y_class_prob = torch.max(x, 0)
        Y_prob = torch.max(Y_class_prob.values)
        Y_hat = torch.argmax(Y_class_prob.values)
        print("Y_class_prob : ", Y_class_prob)
        print("Y_prob : ", Y_prob)
        print("Y_hat : ", Y_hat)
        '''
        

        return Y_prob, Y_hat
    
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
    
    
    
    
