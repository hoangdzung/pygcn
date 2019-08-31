import numpy as np
import scipy.sparse as sp
import torch
import random 
import networkx as nx 

from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
import scipy.sparse as sp
from networkx.utils import UnionFind

def remove_isolate(G):
    origin_n_node = nx.number_of_nodes(G)
    isolate_nodes = []
    for node in G.nodes():
        neighbor = list(G.neighbors(node))
        if node in neighbor:
            neighbor.remove(node)
        if len(neighbor)==0:
            isolate_nodes.append(node)
    G.remove_nodes_from(isolate_nodes)
    print(origin_n_node-nx.number_of_nodes(G), "nodes have been removed")
    return G
# Convert sparse matrix to tuple


def sparse_to_tuple(sparse_mx):
    if not sp.isspmatrix_coo(sparse_mx):
        sparse_mx = sparse_mx.tocoo()
    coords = np.vstack((sparse_mx.row, sparse_mx.col)).transpose()
    values = sparse_mx.data
    shape = sparse_mx.shape
    return coords, values, shape


def minimum_spanning_edges(G, weight='weight', data=False):
    if G.is_directed():
        raise nx.NetworkXError(
            "Mimimum spanning tree not defined for directed graphs.")

    subtrees = UnionFind()
    edges = list(G.edges())
    random.shuffle(edges)
    for u, v in edges:
        if subtrees[u] != subtrees[v]:
            yield (u, v)
            subtrees.union(u, v)

# Perform train-test split
    # Takes in adjacency matrix in sparse format
    # Returns: adj_train, train_edges, val_edges, val_edges_false,
        # test_edges, test_edges_false


def mask_test_edges(adj, test_frac=.1, val_frac=.05, prevent_disconnect=True, verbose=True, seed=42):
    # NOTE: Splits are randomized and results might slightly deviate from
    # reported numbers in the paper.
    if verbose == True:
        print('preprocessing...')
    # Remove diagonal elements
    adj = adj - sp.dia_matrix((adj.diagonal()[np.newaxis, :], [0]), shape=adj.shape)
    adj.eliminate_zeros()
    # Check that diag is zero:
    # assert np.diag(adj.todense()).sum() == 0

    g = nx.from_scipy_sparse_matrix(adj)
    orig_num_cc = nx.number_connected_components(g)

    adj_triu = sp.triu(adj)  # upper triangular portion of adj matrix
    adj_tuple = sparse_to_tuple(adj_triu)  # (coords, values, shape), edges only 1 way
    edges = adj_tuple[0]  # all edges, listed only once (not 2 ways)
    # edges_all = sparse_to_tuple(adj)[0] # ALL edges (includes both ways)
    # controls how large the test set should be
    num_test = int(np.floor(edges.shape[0] * test_frac))
    # controls how alrge the validation set should be
    num_val = int(np.floor(edges.shape[0] * val_frac))
    print("Num edges %d" % (edges.shape[0]))
    print("Test size: %d" % (num_test))
    print("Val size: %d" % (num_val))

    # Store edges in list of ordered tuples (node1, node2) where node1 < node2
    edge_tuples = [(min(edge[0], edge[1]), max(edge[0], edge[1])) for edge in edges]
    all_edge_tuples = set(edge_tuples)

    if verbose == True:
        print('generating test/val sets...')
    # Create a minimum spanning tree
    mst = list(minimum_spanning_edges(g))
    mst_tuples = set([(min(edge[0], edge[1]), max(edge[0], edge[1])) for edge in mst])

    candidate_test_tuples = all_edge_tuples - mst_tuples
    test_edges = set(random.sample(candidate_test_tuples, num_test))
    candidate_val_tuples = candidate_test_tuples - test_edges
    val_edges = set(random.sample(candidate_val_tuples, num_val))
    train_edges = all_edge_tuples - test_edges - val_edges
    g.remove_edges_from(list(test_edges))
    g.remove_edges_from(list(val_edges))

    # # Iterate over shuffled edges, add to train/val sets
    # np.random.shuffle(edge_tuples)
    # for edge in edge_tuples:
    #     # print edge
    #     node1 = edge[0]
    #     node2 = edge[1]

    #     # If removing edge would disconnect a connected component, backtrack and move on
    #     g.remove_edge(node1, node2)
    #     if prevent_disconnect == True:
    #         if nx.number_connected_components(g) > orig_num_cc:
    #             g.add_edge(node1, node2)
    #             continue

    #     # Fill test_edges first
    #     if len(test_edges) < num_test:
    #         test_edges.add(edge)
    #         train_edges.remove(edge)

    #     # Then, fill val_edges
    #     elif len(val_edges) < num_val:
    #         val_edges.add(edge)
    #         train_edges.remove(edge)

    #     # Both edge lists full --> break loop
    #     elif len(test_edges) == num_test and len(val_edges) == num_val:
    #         break
    #     if len(test_edges)%100==0:
    #         print(len(test_edges))

    if (len(val_edges) < num_val or len(test_edges) < num_test):
        print("WARNING: not enough removable edges to perform full train-test split!")
        print("Num. (test, val) edges requested: (" +
              str(num_test) + ", " + str(num_val) + ")")
        print("Num. (test, val) edges returned: (" +
              str(len(test_edges)) + ", " + str(len(val_edges)) + ")")

    if prevent_disconnect == True:
        assert nx.number_connected_components(g) == orig_num_cc

    if verbose == True:
        print('creating false test edges...')

    test_edges_false = set()
    while len(test_edges_false) < num_test:
        idx_i = np.random.randint(0, adj.shape[0])
        idx_j = np.random.randint(0, adj.shape[0])
        if idx_i == idx_j:
            continue

        false_edge = (min(idx_i, idx_j), max(idx_i, idx_j))

        # Make sure false_edge not an actual edge, and not a repeat
        if false_edge in all_edge_tuples:
            continue
        if false_edge in test_edges_false:
            continue

        test_edges_false.add(false_edge)

    if verbose == True:
        print('creating false val edges...')

    val_edges_false = set()
    while len(val_edges_false) < num_val:
        idx_i = np.random.randint(0, adj.shape[0])
        idx_j = np.random.randint(0, adj.shape[0])
        if idx_i == idx_j:
            continue

        false_edge = (min(idx_i, idx_j), max(idx_i, idx_j))

        # Make sure false_edge in not an actual edge, not in test_edges_false, not
        # a repeat
        if false_edge in all_edge_tuples or \
                false_edge in test_edges_false or \
                false_edge in val_edges_false:
            continue

        val_edges_false.add(false_edge)

    if verbose == True:
        print('creating false train edges...')

    train_edges_false = set()
    while len(train_edges_false) < len(train_edges):
        idx_i = np.random.randint(0, adj.shape[0])
        idx_j = np.random.randint(0, adj.shape[0])
        if idx_i == idx_j:
            continue

        false_edge = (min(idx_i, idx_j), max(idx_i, idx_j))

        # Make sure false_edge in not an actual edge, not in test_edges_false,
        # not in val_edges_false, not a repeat
        if false_edge in all_edge_tuples or \
                false_edge in test_edges_false or \
                false_edge in val_edges_false or \
                false_edge in train_edges_false:
            continue

        train_edges_false.add(false_edge)

    if verbose == True:
        print('final checks for disjointness...')

    # assert: false_edges are actually false (not in all_edge_tuples)
    assert test_edges_false.isdisjoint(all_edge_tuples)
    assert val_edges_false.isdisjoint(all_edge_tuples)
    assert train_edges_false.isdisjoint(all_edge_tuples)

    # assert: test, val, train false edges disjoint
    assert test_edges_false.isdisjoint(val_edges_false)
    assert test_edges_false.isdisjoint(train_edges_false)
    assert val_edges_false.isdisjoint(train_edges_false)

    # assert: test, val, train positive edges disjoint
    assert val_edges.isdisjoint(train_edges)
    assert test_edges.isdisjoint(train_edges)
    assert val_edges.isdisjoint(test_edges)

    if verbose == True:
        print('creating adj_train...')

    # Re-build adj matrix using remaining graph
    adj_train = nx.adjacency_matrix(g)

    # Convert edge-lists to numpy arrays
    train_edges = np.array([list(edge_tuple) for edge_tuple in train_edges])
    train_edges_false = np.array([list(edge_tuple) for edge_tuple in train_edges_false])
    val_edges = np.array([list(edge_tuple) for edge_tuple in val_edges])
    val_edges_false = np.array([list(edge_tuple) for edge_tuple in val_edges_false])
    test_edges = np.array([list(edge_tuple) for edge_tuple in test_edges])
    test_edges_false = np.array([list(edge_tuple) for edge_tuple in test_edges_false])

    if verbose == True:
        print('Done with train-test split!')
        print('')
    # NOTE: these edge lists only contain single direction of edge!
    return adj_train, train_edges, train_edges_false, \
        val_edges, val_edges_false, test_edges, test_edges_false


def get_roc_score(adj_sparse, edges_pos, edges_neg, embeddings, with_neg=True):
    # score_matrix = np.dot(embeddings, embeddings.T)

    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    def score_matrix(x, y):
        return (embeddings[x] * embeddings[y]).sum()

    # Store positive edge predictions, actual values
    preds_pos = []
    pos = []
    for edge in edges_pos:
        preds_pos.append(sigmoid(score_matrix(edge[0], edge[1])))  # predicted score
        pos.append(adj_sparse[edge[0], edge[1]])  # actual value (1 for positive)

    # Store negative edge predictions, actual values
    preds_neg = []
    neg = []
    if with_neg:
        for edge in edges_neg:
            preds_neg.append(sigmoid(score_matrix(edge[0], edge[1])))  # predicted score
            neg.append(adj_sparse[edge[0], edge[1]])  # actual value (0 for negative)

    # Calculate scores
    preds_all = np.hstack([preds_pos, preds_neg])
    labels_all = np.hstack([np.ones(len(preds_pos)), np.zeros(len(preds_neg))])
    if with_neg:
        roc_score = roc_auc_score(labels_all, preds_all)
        ap_score = average_precision_score(labels_all, preds_all)
    else:
        pdb.set_trace()
    return roc_score, ap_score

def fixed_unigram_candidate_sampler(num_sampled, unique, range_max, distortion, unigrams):
    weights = unigrams**distortion
    prob = weights/weights.sum()
    sampled = np.random.choice(range_max, num_sampled, p=prob, replace=~unique)
    return sampled

def sample_negative(anchors, neg_adj_list):
    return [random.choice(neg_adj_list[anchor]) for anchor in anchors]
        
def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                    enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)),
                             dtype=np.int32)
    return labels_onehot


def load_data(path="../data/cora/", dataset="cora"):
    """Load citation network dataset (cora only for now)"""
    print('Loading {} dataset...'.format(dataset))

    idx_features_labels = np.genfromtxt("{}{}.content".format(path, dataset),
                                        dtype=np.dtype(str))
    features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
    labels = encode_onehot(idx_features_labels[:, -1])

    # build graph
    idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
    idx_map = {j: i for i, j in enumerate(idx)}
    edges_unordered = np.genfromtxt("{}{}.cites".format(path, dataset),
                                    dtype=np.int32)
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
                     dtype=np.int32).reshape(edges_unordered.shape)
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(labels.shape[0], labels.shape[0]),
                        dtype=np.float32)

    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    deg = adj.sum(0)
    features = normalize(features)
    adj = normalize(adj + sp.eye(adj.shape[0]))

    idx_train = range(140)
    idx_val = range(200, 500)
    idx_test = range(500, 1500)

    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(np.where(labels)[1])
    adj_sp = sparse_mx_to_torch_sparse_tensor(adj)
    adj_ds = torch.FloatTensor(adj.toarray())

    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    neg_adj_list = [list( np.where(adj_ds[i] ==0)[0] ) for i in range(adj_ds.shape[0])]

    return adj_sp, adj_ds, neg_adj_list, edges, np.squeeze(np.array(deg)), features, labels, idx_train, idx_val, idx_test


def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)
