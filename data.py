import torch, os
import numpy as np 
import scipy.sparse as sp
import torch_geometric.datasets as geo_data

DATA_ROOT = 'data'

def load_data(data_name='cora', normalize_feature=True, cuda=False):
    # can use other dataset, some doesn't have mask
    data = geo_data.Planetoid(os.path.join(DATA_ROOT, data_name), data_name).data 
    n = len(data.x)
    # get adjacency matrix
    adj = sp.csr_matrix((np.ones(data.edge_index.shape[1]), data.edge_index), shape=(n,n))
    adj = adj + adj.T - adj.multiply(adj.T) + sp.eye(adj.shape[0])
    adj = normalize_adj(adj)
    data.adj = to_torch_sparse(adj)
    # normalize feature
    if normalize_feature:
        data.x = row_l1_normalize(data.x)

    data.train_mask = data.train_mask.type(torch.bool)
    data.val_mask = data.val_mask.type(torch.bool)
    data.test_mask = data.test_mask.type(torch.bool)
    if cuda:
        data.x = data.x.cuda()
        data.y = data.y.cuda()
        data.adj = data.adj.cuda()
    return data

def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    # add self-loop and normalization also affects performance a lot 
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()

def normalize_adj_row(adj):
    """Row-normalize sparse matrix"""
    rowsum = np.array(adj.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(adj)
    return mx 

def to_torch_sparse(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

def row_l1_normalize(X):
    norm = 1e-6 + X.sum(dim=1, keepdim=True)
    return X/norm

if __name__ == "__main__":
    import sys
    print(sys.version)
    # test goes here
    data = load_data(cuda=True)
    print(data.train_mask[:150])