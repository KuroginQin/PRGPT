# Evaluate the online inference of PR-GPT on the static GP benchmark

import argparse
from modules.X1 import *
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components

from infomap import Infomap
from sdp_clustering import leiden_locale, init_random_seed

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


def get_init_res(ind_est, edges):
    '''
    Functon to get a feasible GP result based on the model outputs
    :param ind_est: node pair estimated probability
    :param edges: edge list
    :return:
    '''
    # ====================
    num_nodes = np.max(np.max(edges)) + 1
    num_edges = len(edges)
    psv_src_idxs = []
    psv_dst_idxs = []
    psv_vals = []
    for t in range(num_edges):
        if ind_est[t] >= 0.5:
            (src, dst) = edges[t]
            # ==========
            psv_src_idxs.append(src)
            psv_dst_idxs.append(dst)
            psv_vals.append(1.0)
            # ==========
            psv_src_idxs.append(dst)
            psv_dst_idxs.append(src)
            psv_vals.append(1.0)
    adj_sp = sp.csr_matrix((psv_vals, (psv_src_idxs, psv_dst_idxs)))
    # ==========
    # Extract clusters/communities w.r.t. connected components
    num_clus_est, clus_res_ = connected_components(csgraph=adj_sp, directed=False, return_labels=True)
    clus_res_ = list(clus_res_)
    comp_mem_cnt = [0 for _ in range(num_clus_est)]
    for i in range(len(clus_res_)):
        comp_mem_cnt[clus_res_[i]] += 1
    clus_res = [-(i+1) for i in range(num_nodes)]
    for i in range(len(clus_res_)):
        if comp_mem_cnt[clus_res_[i]] > 1:
            clus_res[i] = clus_res_[i]
    print('#NODE %d #SUP-NODE %d' % (num_nodes, num_clus_est))
    if num_clus_est == 1:
        return clus_res, None, None, None, 1
    # ==========
    init_node_idx = 0
    init_node_map = {}
    init_edge_map = {}
    for (src, dst) in edges:
        src_lbl = clus_res[src]
        dst_lbl = clus_res[dst]
        #if src_lbl == dst_lbl: continue
        # ==========
        if src_lbl not in init_node_map:
            src_node_idx = init_node_idx
            init_node_map[src_lbl] = init_node_idx
            init_node_idx += 1
        else:
            src_node_idx = init_node_map[src_lbl]
        # ==========
        if dst_lbl not in init_node_map:
            dst_node_idx = init_node_idx
            init_node_map[dst_lbl] = init_node_idx
            init_node_idx += 1
        else:
            dst_node_idx = init_node_map[dst_lbl]
        # ==========
        if src_node_idx > dst_node_idx:
            tmp = src_node_idx
            src_node_idx = dst_node_idx
            dst_node_idx = tmp
        if (src_node_idx, dst_node_idx) not in init_edge_map:
            init_edge_map[(src_node_idx, dst_node_idx)] = 1.0
        else:
            init_edge_map[(src_node_idx, dst_node_idx)] += 1.0
    # ==========
    init_edges = [(src, dst, init_edge_map[(src, dst)]) for (src, dst) in init_edge_map]
    #init_edges = sorted(init_edges)
    init_src_idxs = []
    init_dst_idxs = []
    init_vals = []
    for (src, dst, val) in init_edges:
        init_src_idxs.append(src)
        init_dst_idxs.append(dst)
        init_vals.append(val)
        if src != dst:
            init_src_idxs.append(dst)
            init_dst_idxs.append(src)
            init_vals.append(val)
    graph = sp.coo_matrix((init_vals, (init_src_idxs, init_dst_idxs)))

    return clus_res, graph, init_edges, init_node_map, init_node_idx


def InfoMap_rfn(init_edges, init_node_map, init_num_nodes, clus_res, num_nodes):
    '''
    Function of online refinement via InfoMap
    :param init_edges:
    :param init_node_map:
    :param init_num_nodes:
    :param clus_res:
    :param num_nodes:
    :return:
    '''
    # ====================
    im = Infomap(silent=True)
    for (src, dst, wei) in init_edges:
        im.add_link(src, dst, wei)
    im.run()
    # ==========
    rfn_res_ = [-1 for _ in range(init_num_nodes)]
    for node in im.tree:
        if node.is_leaf:
            rfn_res_[node.node_id] = node.module_id - 1
    rfn_res = [rfn_res_[init_node_map[clus_res[i]]] for i in range(num_nodes)]

    return rfn_res


def locale_rfn(graph, init_node_map, clus_res, num_nodes, rand_seed):
    '''
    Function of online refinement via Locale
    :param graph:
    :param init_node_map:
    :param clus_res:
    :param num_nodes:
    :param rand_seed:
    :return:
    '''
    init_random_seed(rand_seed)
    rfn_res_ = leiden_locale(graph, k=8, eps=1e-6, max_outer=10, max_lv=10, max_inner=2, verbose=0)
    rfn_res = [rfn_res_[init_node_map[clus_res[i]]] for i in range(num_nodes)]

    return rfn_res


def clus_reorder(num_nodes, clus_res):
    '''
    Function to reorder the label assignment of a GP result
    :param num_nodes: number of nodes
    :param clus_res: label assignment to be reordered
    :return:
    '''
    clus_res_ = []
    lbl_cnt = 0
    lbl_map = {}
    for i in range(num_nodes):
        lbl = clus_res[i]
        if lbl not in lbl_map:
            lbl_map[lbl] = lbl_cnt
            clus_res_.append(lbl_cnt)
            lbl_cnt += 1
        else:
            clus_res_.append(lbl_map[lbl])

    return clus_res_, lbl_cnt


if __name__ == '__main__':
    # ====================
    setup_seed(rand_seed_gbl)

    # ====================
    parser = argparse.ArgumentParser()
    parser.add_argument('--N', type=int, default=100000)
    parser.add_argument('--eph', type=int, default=4) # 1-100
    args = parser.parse_args()
    # ==========
    tst_num_nodes = args.N # Number of nodes
    eph_idx = args.eph # Index of epoch w.r.t. the pre-trained model
    tst_num_snap = 5 # Number of snapshots
    mu = 2.5 # ratio_within_over_between
    beta = 3 # block_size_heterogeneity

    # ====================
    # Layer configurations & parameter settings
    emb_dim = 32 # Embedding dimensionality
    num_feat_lyr = 2 # Number of MLP layers in feat extraction module
    num_GNN_lyr = 2 # Number of GNN layers
    num_MLP_lyr_tmp = 4 # Number of MLP layers in binary classifier

    # ====================
    # Load the dataset
    pkl_file = open('data/static_%d_%d_%.1f_%.1f_edges_list.pickle'
                    % (tst_num_snap, tst_num_nodes, mu, beta), 'rb')
    tst_edges_list = pickle.load(pkl_file)
    pkl_file.close()
    # ==========
    pkl_file = open('data/static_%d_%d_%.1f_%.1f_gnd_list.pickle'
                    % (tst_num_snap, tst_num_nodes, mu, beta), 'rb')
    tst_gnd_list = pickle.load(pkl_file)
    pkl_file.close()

    # ====================
    # Load the pre-trained model
    mdl = MDL_X1(emb_dim, num_feat_lyr, num_GNN_lyr, num_MLP_lyr_tmp, drop_rate=0.0).to(device)
    mdl.load_state_dict(torch.load('chpt/X1_mdl_%d.pt' % (eph_idx)))
    mdl.eval()

    # ====================
    # Pre-compute some topo stat
    tst_degs_list = []
    tst_src_idxs_list = []
    tst_dst_idxs_list = []
    tst_edges_inf_list = []
    for t in range(tst_num_snap):
        # ==========
        tst_edges = tst_edges_list[t]
        if np.min(tst_edges) == 1:
            zero_base = False
        else:
            zero_base = True
        if not zero_base:
            tst_edges = [(src-1, dst-1) for (src, dst) in tst_edges]
            tst_edges_list[t] = tst_edges
        # ==========
        tst_gnd = tst_gnd_list[t]
        if int(np.min(tst_gnd)) > 0:
            tst_gnd = tst_gnd - 1
            tst_gnd_list[t] = tst_gnd
        # ==========
        tst_degs = [0 for _ in range(tst_num_nodes)]
        tst_src_idxs = []
        tst_dst_idxs = []
        for (src, dst) in tst_edges:
            # ==========
            tst_degs[src] += 1
            tst_degs[dst] += 1
            # ==========
            tst_src_idxs.append(src)
            tst_dst_idxs.append(dst)
        tst_edges_inf = tst_edges.copy()
        # ==========
        tst_degs_list.append(tst_degs)
        tst_src_idxs_list.append(tst_src_idxs)
        tst_dst_idxs_list.append(tst_dst_idxs)
        tst_edges_inf_list.append(tst_edges_inf)

    # ====================
    # Evaluate the online inference of PR-GPT over 5 independent runs
    AC_IM_list = []
    ARI_IM_list = []
    F1_IM_list = []
    RCL_IM_list = []
    PCN_IM_list = []
    # ==========
    AC_Lcl_list = []
    ARI_Lcl_list = []
    F1_Lcl_list = []
    RCL_Lcl_list = []
    PCN_Lcl_list = []
    # ==========
    feat_time_list = []
    FFP_time_list = []
    init_time_list = []
    rfn_time_IM_list = []
    rfn_time_Lcl_list = []
    time_IM_list = []
    time_Lcl_list = []
    # ==========
    for t in range(tst_num_snap):
        # ==========
        tst_edges = tst_edges_list[t]
        tst_num_edges = len(tst_edges)
        tst_gnd = tst_gnd_list[t]
        tst_num_clus = np.max(tst_gnd) + 1
        # ==========
        tst_degs = tst_degs_list[t]
        tst_degs_tnr = torch.FloatTensor(tst_degs).to(device)
        # ==========
        tst_src_idxs = tst_src_idxs_list[t]
        tst_dst_idxs = tst_dst_idxs_list[t]
        tst_edges_inf = tst_edges_inf_list[t]

        # ====================
        idxs, vals = get_sp_adj(tst_edges)
        idxs_tnr = torch.LongTensor(idxs).to(device)
        vals_tnr = torch.FloatTensor(vals).to(device)
        ptn_sp_adj_tnr = torch.sparse.FloatTensor(idxs_tnr.t(), vals_tnr,
                                                  torch.Size([tst_num_nodes, tst_num_nodes])).to(device)
        # ==========
        idxs, vals = get_sp_GCN_sup(tst_edges, tst_degs)
        idxs_tnr = torch.LongTensor(idxs).to(device)
        vals_tnr = torch.FloatTensor(vals).to(device)
        sup_tnr = torch.sparse.FloatTensor(idxs_tnr.t(), vals_tnr,
                                           torch.Size([tst_num_nodes,
                                                       tst_num_nodes])).to(device)

        # ====================
        # Feat ext via Gaussian rand proj
        time_s = time.time()
        # ==========
        rand_mat = get_rand_proj_mat(tst_num_nodes, emb_dim, rand_seed=rand_seed_gbl)
        rand_mat_tnr = torch.FloatTensor(rand_mat).to(device)
        red_feat_tnr = get_red_feat(ptn_sp_adj_tnr,
                                    torch.reshape(tst_degs_tnr, (-1, 1)),
                                    rand_mat_tnr, tst_num_edges)
        # ==========
        time_e = time.time()
        feat_time = time_e - time_s
        feat_time_list.append(feat_time)
        # ==========
        del ptn_sp_adj_tnr, rand_mat_tnr

        # ====================
        # One FFP of the model
        time_s = time.time()
        # ==========
        emb_tnr, lft_tmp_tnr, rgt_tmp_tnr = mdl(red_feat_tnr, sup_tnr)
        edge_ind_est = get_edge_ind_est(emb_tnr, lft_tmp_tnr, rgt_tmp_tnr, tst_src_idxs, tst_dst_idxs)
        # ==========
        time_e = time.time()
        FFP_time = time_e - time_s
        FFP_time_list.append(FFP_time)
        del emb_tnr, lft_tmp_tnr, rgt_tmp_tnr

        # ====================
        if torch.cuda.is_available():
            edge_ind_est = edge_ind_est.cpu().data.numpy()
        else:
            edge_ind_est = edge_ind_est.data.numpy()

        # ====================
        # Result derivation
        time_s = time.time()
        # ==========
        clus_res_init, init_graph, init_edges, init_node_map, init_num_nodes = \
            get_init_res(edge_ind_est, tst_edges_inf)
        # ==========
        time_e = time.time()
        init_time = time_e - time_s
        init_time_list.append(init_time)
        # ==========
        if init_num_nodes == 1:
            break

        # =====================
        # Online refinement via InfoMap
        time_s = time.time()
        clus_res_IM = InfoMap_rfn(init_edges, init_node_map, init_num_nodes, clus_res_init, tst_num_nodes)
        time_e = time.time()
        rfn_time_IM = time_e - time_s
        rfn_time_IM_list.append(rfn_time_IM)
        # =====================
        # Online refinement via Locale
        time_s = time.time()
        clus_res_Lcl = locale_rfn(init_graph, init_node_map, clus_res_init, tst_num_nodes, rand_seed=0)
        time_e = time.time()
        rfn_time_Lcl = time_e - time_s
        rfn_time_Lcl_list.append(rfn_time_Lcl)

        # ====================
        # Evaluation for PR-GPT w/ InfoMap
        time_IM = feat_time + FFP_time + init_time + rfn_time_IM
        time_IM_list.append(time_IM)
        # ==========
        clus_res_IM, num_clus_IM = clus_reorder(tst_num_nodes, clus_res_IM)
        AC_IM, ARI_IM, RCL_IM, PCN_IM = quality_eva(tst_gnd, clus_res_IM)
        F1_IM = 2*RCL_IM*PCN_IM / (PCN_IM+RCL_IM)
        AC_IM_list.append(AC_IM)
        ARI_IM_list.append(ARI_IM)
        RCL_IM_list.append(RCL_IM)
        PCN_IM_list.append(PCN_IM)
        F1_IM_list.append(F1_IM)
        # ==========
        print('SNAP-%d IM Gnd-K %d Alg-K %d '
              'AC %.4f ARI %.4f F1 %.4f (RCL %.4f PCN %.4f) '
              'TIME %.4f (%.4f %.4f %.4f %.4f)'
              % (t+1, tst_num_clus, num_clus_IM,
                 AC_IM, ARI_IM, F1_IM, RCL_IM, PCN_IM,
                 time_IM, feat_time, FFP_time, init_time, rfn_time_IM))

        # ====================
        # Evaluation for PR-GPT w/ Locale
        time_Lcl = feat_time + FFP_time + init_time + rfn_time_Lcl
        time_Lcl_list.append(time_Lcl)
        # ==========
        clus_res_Lcl, num_clus_Lcl = clus_reorder(tst_num_nodes, clus_res_Lcl)
        AC_Lcl, ARI_Lcl, RCL_Lcl, PCN_Lcl = quality_eva(tst_gnd, clus_res_Lcl)
        F1_Lcl = 2*RCL_Lcl*PCN_Lcl / (PCN_Lcl+RCL_Lcl)
        AC_Lcl_list.append(AC_Lcl)
        ARI_Lcl_list.append(ARI_Lcl)
        RCL_Lcl_list.append(RCL_Lcl)
        PCN_Lcl_list.append(PCN_Lcl)
        F1_Lcl_list.append(F1_Lcl)
        # ==========
        print('SNAP-%d Lcl Gnd-K %d Alg-K %d '
              'AC %.4f ARI %.4f F1 %.4f (RCL %.4f PCN %.4f) '
              'TIME %.4f (%.4f %.4f %.4f %.4f)'
              % (t+1, tst_num_clus, num_clus_Lcl,
                 AC_Lcl, ARI_Lcl, F1_Lcl, RCL_Lcl, PCN_Lcl,
                 time_Lcl, feat_time, FFP_time, init_time, rfn_time_Lcl))

    # ====================
    # Average evaluation results for PR-GPT w/ InfoMap
    feat_mean = np.mean(feat_time_list)
    feat_std = np.std(feat_time_list)
    FFP_mean = np.mean(FFP_time_list)
    FFP_std = np.std(FFP_time_list)
    init_mean = np.mean(init_time_list)
    init_std = np.std(init_time_list)
    # ====================
    AC_IM_mean = np.mean(AC_IM_list)
    AC_IM_std = np.std(AC_IM_list)
    ARI_IM_mean = np.mean(ARI_IM_list)
    ARI_IM_std = np.std(ARI_IM_list)
    F1_IM_mean = np.mean(F1_IM_list)
    F1_IM_std = np.std(F1_IM_list)
    RCL_IM_mean = np.mean(RCL_IM_list)
    RCL_IM_std = np.std(RCL_IM_list)
    PCN_IM_mean = np.mean(PCN_IM_list)
    PCN_IM_std = np.std(PCN_IM_list)
    rfn_IM_mean = np.mean(rfn_time_IM_list)
    rfn_IM_std = np.std(rfn_time_IM_list)
    time_IM_mean = np.mean(time_IM_list)
    time_IM_std = np.std(time_IM_list)
    print('====================')
    print('IM AC %.4f %.4f ARI %.4f %.4f F1 %.4f %.4f '
          '(RCL %.4f %.4f PCN %.4f %.4f)' %
          (AC_IM_mean, AC_IM_std, ARI_IM_mean, ARI_IM_std, F1_IM_mean, F1_IM_std,
           RCL_IM_mean, RCL_IM_std, PCN_IM_mean, PCN_IM_std))
    print('IM TIME %.4f %.4f FEAT %.4f %.4f FFP %.4f %.4f INIT %.4f %.4f RFN %.4f %.4f' %
          (time_IM_mean, time_IM_std, feat_mean, feat_std,
           FFP_mean, FFP_std, init_mean, init_std, rfn_IM_mean, rfn_IM_std))

    # ====================
    # Average evaluation results for PR-GPT w/ Locale
    AC_Lcl_mean = np.mean(AC_Lcl_list)
    AC_Lcl_std = np.std(AC_Lcl_list)
    ARI_Lcl_mean = np.mean(ARI_Lcl_list)
    ARI_Lcl_std = np.std(ARI_Lcl_list)
    F1_Lcl_mean = np.mean(F1_Lcl_list)
    F1_Lcl_std = np.std(F1_Lcl_list)
    RCL_Lcl_mean = np.mean(RCL_Lcl_list)
    RCL_Lcl_std = np.std(RCL_Lcl_list)
    PCN_Lcl_mean = np.mean(PCN_Lcl_list)
    PCN_Lcl_std = np.std(PCN_Lcl_list)
    rfn_Lcl_mean = np.mean(rfn_time_Lcl_list)
    rfn_Lcl_std = np.std(rfn_time_Lcl_list)
    time_Lcl_mean = np.mean(time_Lcl_list)
    time_Lcl_std = np.std(time_Lcl_list)
    print('====================')
    print('Lcl AC %.4f %.4f ARI %.4f %.4f F1 %.4f %.4f '
          '(RCL %.4f %.4f PCN %.4f %.4f)' %
          (AC_Lcl_mean, AC_Lcl_std, ARI_Lcl_mean, ARI_Lcl_std, F1_Lcl_mean, F1_Lcl_std,
           RCL_Lcl_mean, RCL_Lcl_std, PCN_Lcl_mean, PCN_Lcl_std))
    print('Lcl TIME %.4f %.4f FEAT %.4f %.4f FFP %.4f %.4f INIT %.4f %.4f RFN %.4f %.4f' %
          (time_Lcl_mean, time_Lcl_std, feat_mean, feat_std,
           FFP_mean, FFP_std, init_mean, init_std, rfn_Lcl_mean, rfn_Lcl_std))
    print()

