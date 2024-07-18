# Offline pre-training of PR-GPT on small synthetic graphs

import argparse
from modules.X1 import *
import torch.optim as optim
import scipy.sparse as sp

import pickle
import random
import time
from utils import *

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
rand_seed_gbl = 0

torch.cuda.set_device(0)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_rand_proj_mat(data_dim, red_dim, rand_seed=None):
    '''
    Function to get random matrix for Gaussian random projection
    :param data_dim: original data dimensionality
    :param red_dim: reduced dimensionality
    :param rand_seed: random seed
    :return: random matrix
    '''
    # ===================
    np.random.seed(rand_seed)
    rand_mat = np.random.normal(0, 1.0/red_dim, (data_dim, red_dim))

    return rand_mat


def get_sp_adj(edges):
    '''
    Function to get sparse adj mat
    :param edges: edge list
    :return: idxs: list of node pair; vals: corresponding edge weights
    '''
    # ====================
    idxs = []
    vals = []
    for (src, dst) in edges:
        idxs.append((src, dst))
        vals.append(1)
        idxs.append((dst, src))
        vals.append(1)

    return idxs, vals


def get_sp_GCN_sup(edges, degs):
    '''
    Function to get GCN support (i.e., normalized adj mat w/ self-edges)
    :param edges: edge list
    :param degs: list of node degrees
    :return: idxs: list of node pair; vals: corresponding weights
    '''
    # ====================
    num_nodes = len(degs)
    degs = [(deg + 1) for deg in degs]
    degs_sq = np.sqrt(degs)
    # ==========
    idxs = []
    vals = []
    for (src, dst) in edges:
        # ==========
        v = 1/(degs_sq[src] * degs_sq[dst])
        # ==========
        idxs.append((src, dst))
        vals.append(v)
        idxs.append((dst, src))
        vals.append(v)
    for idx in range(num_nodes): # Self-edges
        idxs.append((idx, idx))
        vals.append(1/degs[idx])

    return idxs, vals


def get_red_feat(sp_adj_tnr, degs_tnr, rand_mat_tnr, num_edges):
    '''
    Function to get reduced community-preserving feat via Gaussian random projection
    :param sp_adj_tnr: sparse adj mat
    :param degs_tnr: vector of node deg
    :param rand_mat_tnr: Gaussian rand mat
    :param num_edges: number of edges
    :return: low-dim community-psv feat
    '''
    return torch.spmm(sp_adj_tnr, rand_mat_tnr) \
           - torch.mm(degs_tnr, torch.mm(degs_tnr.t(), rand_mat_tnr)) / (2*num_edges)


if __name__ == '__main__':
    # ====================
    setup_seed(rand_seed_gbl)

    # ====================
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_pth', type=str, default='./chpt_new')
    parser.add_argument('--alpha', type=float, default=0.1)
    parser.add_argument('--lambd', type=float, default=10)
    parser.add_argument('--drop_rate', type=float, default=0.2)
    parser.add_argument('--learn_rate', type=float, default=5e-5)
    parser.add_argument('--num_ephs', type=int, default=100)
    args = parser.parse_args()
    # ==========
    save_pth = args.save_pth # Path to save model parameters
    BCE_param = args.alpha # Hyperparameter - alpha
    mod_rsl = args.lambd # Hyperparameter - lambda
    drop_rate = args.drop_rate # Dropout rate
    learn_rate = args.learn_rate # Learning rate
    num_eph = args.num_ephs # Total number of training epochs

    # ====================
    # Layer configurations & parameter settings
    emb_dim = 32 # Embedding dimensionality
    num_feat_lyr = 2 # Number of MLP layers in feat extraction module
    num_GNN_lyr = 2 # Number of GNN layers
    num_MLP_lyr_tmp = 4 # Number of MLP layers in binary classifier

    # ====================
    # Load pre-training data
    pkl_file = open('data/ptns_edges_list.pickle', 'rb')
    ptn_edges_list = pickle.load(pkl_file)
    pkl_file.close()
    # ==========
    pkl_file = open('data/ptns_gnd_list.pickle', 'rb')
    ptn_gnd_list = pickle.load(pkl_file)
    pkl_file.close()
    # ==========
    ptn_num_snap = len(ptn_edges_list) # Number of (synthetic) snapshots/graphs for pre-training

    # ====================
    # Pre-compute topo stat
    ptn_num_nodes_list = []
    ptn_gnd_mem_list = []
    ptn_degs_list = []
    ptn_src_idxs_list = []
    ptn_dst_idxs_list = []
    for t in range(ptn_num_snap):
        # ==========
        ptn_edges = ptn_edges_list[t]
        ptn_gnd = ptn_gnd_list[t]  # Ground-truth
        # ==========
        ptn_num_nodes = len(ptn_gnd)
        ptn_num_nodes_list.append(ptn_num_nodes)
        # ==========
        if np.min(ptn_gnd) == 0:
            ptn_num_clus = np.max(ptn_gnd) + 1
        else:
            ptn_num_clus = np.max(ptn_gnd)
        ptn_gnd_mem = np.zeros((ptn_num_nodes, ptn_num_clus))
        for i in range(ptn_num_nodes):
            r = ptn_gnd[i]
            ptn_gnd_mem[i, r] = 1.0
        ptn_gnd_mem_list.append(ptn_gnd_mem)
        # ==========
        ptn_degs = [0 for _ in range(ptn_num_nodes)]
        ptn_src_idxs = []
        ptn_dst_idxs = []
        for (src, dst) in ptn_edges:
            # ==========
            ptn_degs[src] += 1
            ptn_degs[dst] += 1
            # ==========
            ptn_src_idxs.append(src)
            ptn_dst_idxs.append(dst)
        ptn_degs_list.append(ptn_degs)
        ptn_src_idxs_list.append(ptn_src_idxs)
        ptn_dst_idxs_list.append(ptn_dst_idxs)

    # ====================
    # Define the model
    mdl = MDL_X1(emb_dim, num_feat_lyr, num_GNN_lyr, num_MLP_lyr_tmp, drop_rate).to(device)
    # ==========
    # Define the optimizer
    opt = optim.Adam(mdl.parameters(), lr=learn_rate)

    # ====================
    for eph in range(num_eph):
        # ====================
        mdl.train()
        loss_acc = 0.0
        for t in range(ptn_num_snap):
            # ====================
            ptn_num_nodes = ptn_num_nodes_list[t]
            ptn_edges = ptn_edges_list[t] # Edge list
            ptn_num_edges = len(ptn_edges)
            # ==========
            ptn_degs = ptn_degs_list[t]
            ptn_degs_tnr = torch.FloatTensor(ptn_degs).to(device)
            # ==========
            ptn_gnd_mem = ptn_gnd_mem_list[t]
            ptn_gnd_mem_tnr = torch.FloatTensor(ptn_gnd_mem).to(device)
            pair_ind_gnd = torch.mm(ptn_gnd_mem_tnr, ptn_gnd_mem_tnr.t()) # Ground-truth pairwise constraint
            # ==========
            ptn_src_idxs = ptn_src_idxs_list[t]
            ptn_dst_idxs = ptn_dst_idxs_list[t]

            # ====================
            # Get GNN support
            idxs, vals = get_sp_GCN_sup(ptn_edges, ptn_degs)
            idxs_tnr = torch.LongTensor(idxs).to(device)
            vals_tnr = torch.FloatTensor(vals).to(device)
            sup_tnr = torch.sparse.FloatTensor(idxs_tnr.t(), vals_tnr,
                                               torch.Size([ptn_num_nodes, ptn_num_nodes])).to(device)

            # ====================
            # Feat ext via Gaussian rand proj
            idxs, vals = get_sp_adj(ptn_edges)
            idxs_tnr = torch.LongTensor(idxs).to(device)
            vals_tnr = torch.FloatTensor(vals).to(device)
            ptn_sp_adj_tnr = torch.sparse.FloatTensor(idxs_tnr.t(), vals_tnr,
                                                      torch.Size([ptn_num_nodes, ptn_num_nodes])).to(device)
            rand_mat = get_rand_proj_mat(ptn_num_nodes, emb_dim, rand_seed=rand_seed_gbl)
            rand_mat_tnr = torch.FloatTensor(rand_mat).to(device)
            red_feat_tnr = get_red_feat(ptn_sp_adj_tnr,
                                        torch.reshape(ptn_degs_tnr, (-1, 1)),
                                        rand_mat_tnr, ptn_num_edges)

            # =====================
            # One FFP of the model & get training loss
            emb_tnr, lft_tmp_tnr, rgt_tmp_tnr = mdl(red_feat_tnr, sup_tnr)
            edge_ind_est = get_edge_ind_est(emb_tnr, lft_tmp_tnr, rgt_tmp_tnr, ptn_src_idxs, ptn_dst_idxs)
            pair_ind_est = get_node_pair_ind_est(emb_tnr, lft_tmp_tnr, rgt_tmp_tnr)
            BCE_loss = get_BCE_loss(pair_ind_gnd, pair_ind_est)
            mod_loss = get_mod_max_loss(edge_ind_est, pair_ind_est, ptn_degs_tnr, ptn_num_edges, resolution=mod_rsl)
            loss = BCE_param*BCE_loss + mod_loss

            # ====================
            opt.zero_grad()
            loss.backward()
            opt.step()
            # ==========
            loss_acc += loss.item()
        print('Epoch %d loss %f' % (eph+1, loss_acc))

        # ====================
        # Save model parameters of current epoch
        torch.save(mdl.state_dict(), '%s/X1_mdl_%d.pt' % (save_pth, eph+1))

