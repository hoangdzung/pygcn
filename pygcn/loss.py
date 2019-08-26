import torch 

def nmin_cut(assign_tensor, adj):
    super_adj = torch.transpose(assign_tensor, 0, 1) @ adj @ assign_tensor
    vol = super_adj.sum(1)
    diag = torch.diagonal(super_adj)
    norm_cut = (vol - diag)/vol
    lozz = norm_cut.sum()
    return lozz