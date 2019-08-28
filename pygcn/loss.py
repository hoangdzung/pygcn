import torch 
import torch.nn.functional as F 

def nmin_cut(assign_tensor, adj):
    super_adj = torch.transpose(assign_tensor, 0, 1) @ adj @ assign_tensor
    vol = super_adj.sum(1)
    diag = torch.diagonal(super_adj)
    norm_cut = (vol - diag)/vol
    lozz = norm_cut.sum()
    return lozz

def sigmoid_cross_entropy_with_logits(labels, logits):
    sig_aff = torch.sigmoid(logits)
    loss = labels * -torch.log(sig_aff+1e-10) + (1 - labels) * -torch.log(1 - sig_aff+1e-10)
    return loss


def node2vec(outputs1, outputs2, neg_outputs, neg_sample_weights=1.0):
    outputs1 = F.normalize(outputs1, dim=1)
    outputs2 = F.normalize(outputs2, dim=1)
    neg_outputs = F.normalize(neg_outputs, dim=1)

    true_aff = F.cosine_similarity(outputs1, outputs2)
    neg_aff = outputs1.mm(neg_outputs.t())    
    true_labels = torch.ones(true_aff.shape)
    if torch.cuda.is_available():
        true_labels = true_labels.cuda()
        true_xent = sigmoid_cross_entropy_with_logits(labels=true_labels, logits=true_aff)
    neg_labels = torch.zeros(neg_aff.shape)
    if torch.cuda.is_available():
        neg_labels = neg_labels.cuda()
    neg_xent = sigmoid_cross_entropy_with_logits(labels=neg_labels, logits=neg_aff)
    loss = true_xent.sum() + neg_sample_weights * neg_xent.sum()
    return loss
