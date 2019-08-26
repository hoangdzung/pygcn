import torch.nn as nn
import torch.nn.functional as F
from pygcn.layers import GraphConvolution
from pygcn.gumbel import gumbel_softmax

class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GCN, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nclass)
        self.dropout = dropout

    def forward(self, x, adj, temp=1, hard=False, beta=0):
        x = self.gc1(x, adj)
        self.params = x 
        embedding_assign = gumbel_softmax(x, temp, hard, beta)
        x = F.relu(x)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        return embedding_assign, F.log_softmax(x, dim=1)
