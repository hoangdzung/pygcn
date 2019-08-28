from __future__ import division
from __future__ import print_function

import time
import argparse
import numpy as np

import torch
import torch.nn.functional as F
import torch.optim as optim

from pygcn.utils import load_data, accuracy, fixed_unigram_candidate_sampler
from pygcn.models import GCN
from pygcn.loss import nmin_cut, node2vec
from pygcn.classify import classify

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--fastmode', action='store_true', default=False,
                    help='Validate during training pass.')
parser.add_argument('--seed', type=int, default=42, 
                    help='Random seed.')

parser.add_argument('--pre_epochs', type=int, default=200,
                    help='Number of epochs to pre-train.')
parser.add_argument('--epochs', type=int, default=200,
                    help='Number of epochs to train.')
parser.add_argument('--batch_size', type=int, default=256,
                    help='Batch size.')
parser.add_argument('--lr', type=float, default=0.01,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=16,
                    help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='Dropout rate (1 - keep probability).')
parser.add_argument('--neg_sample_size',  type=int, default=20,
                    help='Negative sample size')
parser.add_argument('--neg_sample_weight',  type=int, default=20,
                    help='Negative sample size')

parser.add_argument('--transfer', action='store_true', default=False,
                    help='Transfer learning - using smaller learning rate when transfering')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

# Load data
adj, adj_ds, train_edges, degrees, features, labels, idx_train, idx_val, idx_test = load_data()

# Model and optimizer
model = GCN(nfeat=features.shape[1],
            nhid=args.hidden,
            nclass=labels.max().item() + 1,
            dropout=args.dropout)
optimizer = optim.Adam(model.parameters(),
                       lr=args.lr, weight_decay=args.weight_decay)
optimizer2 = optim.Adam(
    [
        {"params": model.gc1.parameters(), "lr": args.lr/10},
        {"params": model.gc2.parameters(), "lr": args.lr}
    ], 
    lr=args.lr, 
    weight_decay=args.weight_decay
)

best_val_acc = 0
best_output = None 
best_at = 0

if args.cuda:
    model.cuda()
    features = features.cuda()
    adj = adj.cuda()
    adj_ds = adj_ds.cuda()
    labels = labels.cuda()
    idx_train = idx_train.cuda()
    idx_val = idx_val.cuda()
    idx_test = idx_test.cuda()


def pretrain(epoch):
    t = time.time()
    model.train()
    optimizer.zero_grad()
    _, _ = model(features, adj)
    neg_nodes = fixed_unigram_candidate_sampler(
        num_sampled=args.neg_sample_size,
        unique=False,
        range_max=len(degrees),
        distortion=0.75,
        unigrams=degrees
    )   
    np.random.shuffle(train_edges)
    batch_edges = train_edges[:args.batch_size]
    nodes1, nodes2 = batch_edges[:,0], batch_edges[:,1]

    outputs1 = F.normalize(model.params[nodes1], dim=1)
    outputs2 = F.normalize(model.params[nodes2], dim=1)
    neg_outputs = F.normalize(model.params[neg_nodes], dim=1)

    loss_train = node2vec(outputs1, outputs2, neg_outputs, args.neg_sample_weight)
#     acc_train = accuracy(output[idx_train], labels[idx_train])
    loss_train.backward()
    optimizer.step()

    if not args.fastmode:
        # Evaluate validation set performance separately,
        # deactivates dropout during validation run.
        model.eval()
        _,_ = model(features, adj)

#     loss_val = F.nll_loss(output[idx_val], labels[idx_val])
#     acc_val = accuracy(output[idx_val], labels[idx_val])
#     print('Epoch: {:04d}'.format(epoch+1),
#          'loss_train: {:.4f}'.format(loss_train.item()),
#          'acc_train: {:.4f}'.format(acc_train.item()),
#           'loss_val: {:.4f}'.format(loss_val.item()),
#          'acc_val: {:.4f}'.format(acc_val.item()),
#          'time: {:.4f}s'.format(time.time() - t))
    if epoch % 100 == 0:
        accs = classify(model.params.detach().cpu().numpy(),labels.detach().cpu().numpy(), 0.5)
        print(loss_train.item(), accs)

def train(epoch):
    global best_val_acc, best_output, best_at

    t = time.time()
    model.train()
    if args.transfer:
        optimizer2.zero_grad()
    else:
        optimizer.zero_grad()

    _, output = model(features, adj)
    loss_train = F.nll_loss(output[idx_train], labels[idx_train])
    acc_train = accuracy(output[idx_train], labels[idx_train])
    loss_train.backward()
    if args.transfer:
        optimizer2.step()
    else:
        optimizer.step()

    if not args.fastmode:
        # Evaluate validation set performance separately,
        # deactivates dropout during validation run.
        model.eval()
        _. output = model(features, adj)

    loss_val = F.nll_loss(output[idx_val], labels[idx_val])
    acc_val = accuracy(output[idx_val], labels[idx_val])
    if acc_val > best_val_acc:
        best_val_acc = acc_val
        best_output = output
        best_at = epoch

    print('Epoch: {:04d}'.format(epoch+1),
          'loss_train: {:.4f}'.format(loss_train.item()),
          'acc_train: {:.4f}'.format(acc_train.item()),
          'loss_val: {:.4f}'.format(loss_val.item()),
          'acc_val: {:.4f}'.format(acc_val.item()),
          'time: {:.4f}s'.format(time.time() - t))


def test(output):
    model.eval()
    loss_test = F.nll_loss(output[idx_test], labels[idx_test])
    acc_test = accuracy(output[idx_test], labels[idx_test])
    print("Test set results:",
          "loss= {:.4f}".format(loss_test.item()),
          "accuracy= {:.4f}".format(acc_test.item()))


# Train model
t_total = time.time()
for epoch in range(args.pre_epochs):
    pretrain(epoch)

for epoch in range(args.epochs):
    train(epoch)
print("Optimization Finished!")
print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

# Testing
print("Best model at ", best_at)
test(best_output)
