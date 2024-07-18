# Evaluate Locale on the static GP benchmark

import argparse
from sdp_clustering import leiden_locale, init_random_seed
import scipy.sparse as sp
import time
import pickle
import random
from utils import *

gbl_rand_seed = 0

if __name__ == '__main__':
    # ====================
    init_random_seed(gbl_rand_seed)

    # ====================
    parser = argparse.ArgumentParser()
    parser.add_argument('--N', type=int, default=100000)
    args = parser.parse_args()
    # ==========
    num_nodes = args.N # Number of nodes
    num_snap = 5 # Number of graphs for each setting
    mu = 2.5 # ratio_within_over_between
    beta = 3 # block_size_heterogeneity

    # ====================
    # Parameter settings of Locale
    EPS = 1e-6 # Stopping criterion for optimization problem
    max_outer = 10 # Maximum number of outer iterations
    max_lv = 10 # Maximum number of levels in an outer iteration
    max_inner = 2 # Maximum number of inner iters for optimization
    verbose = 0 # Verbosity
    k = 8

    # ====================
    # Load the dataset
    edges_list_file = open('data/static_%d_%d_%.1f_%.1f_edges_list.pickle'
                           % (num_snap, num_nodes, mu, beta), 'rb')
    edges_list = pickle.load(edges_list_file)
    edges_list_file.close()
    # ==========
    gnd_list_file = open('data/static_%d_%d_%.1f_%.1f_gnd_list.pickle'
                         % (num_snap, num_nodes, mu, beta), 'rb')
    gnd_list = pickle.load(gnd_list_file)
    gnd_list_file.close()

    # ====================
    acc_list = []
    ARI_list = []
    F1_list = []
    rcl_list = []
    pcn_list = []
    time_list = []
    for t in range(num_snap):
        # ==========
        print('SNAP-#%d' % (t+1))
        edges_ = edges_list[t]
        gnd = gnd_list[t]
        # ==========
        if int(np.min(gnd)) == 0:
            num_clus_gnd = int(np.max(gnd)) + 1 # Number of clusters
        else:
            num_clus_gnd = int(np.max(gnd))
            gnd = gnd - 1

        # ==========
        edges = []
        if np.min(edges_) == 1:
            zero_base = False
        else:
            zero_base = True
        for (src, dst) in edges_:
            if not zero_base: # Ensure that node indices are w/ 0-base
                edges.append((src-1, dst-1))
                edges.append((dst-1, src-1))
            else:
                edges.append((src, dst))
                edges.append((dst, src))
        # ==========
        src_idxs = []
        dst_idxs = []
        vals = []
        for (src, dst) in edges:
            # ==========
            src_idxs.append(src)
            dst_idxs.append(dst)
            vals.append(1.0)
            # ==========
            src_idxs.append(dst)
            dst_idxs.append(src)
            vals.append(1.0)
        graph = sp.coo_matrix((vals, (src_idxs, dst_idxs)))

        # ====================
        time_s = time.time()
        clus_res_ = leiden_locale(graph, k, EPS, max_outer, max_lv, max_inner, verbose)
        time_e = time.time()
        inf_time = time_e - time_s

        # ====================
        lbl_map = {}
        lbl_cnt = 0
        clus_res = []
        for lbl in clus_res_:
            if lbl not in lbl_map:
                lbl_map[lbl] = lbl_cnt
                clus_res.append(lbl_cnt)
                lbl_cnt += 1
            else:
                clus_res.append(lbl_map[lbl])
        num_clus_est = lbl_cnt
        acc, ARI, recall, precision = quality_eva(gnd, clus_res)
        F1 = 2*recall*precision/(precision+recall)
        print('Gnd-K %d Alg-K %d AC %.4f ARI %.4f F1 %.4f (RCL %.4f PCN %.4f) TIME %f'
              % (num_clus_gnd, num_clus_est, acc, ARI,
                 F1, recall, precision, inf_time))
        print()
        # ==========
        acc_list.append(acc)
        ARI_list.append(ARI)
        F1_list.append(F1)
        rcl_list.append(recall)
        pcn_list.append(precision)
        time_list.append(inf_time)

    # ====================
    acc_mean = np.mean(acc_list)
    acc_std = np.std(acc_list)
    ARI_mean = np.mean(ARI_list)
    ARI_std = np.std(ARI_list)
    F1_mean = np.mean(F1_list)
    F1_std = np.std(F1_list)
    rcl_mean = np.mean(rcl_list)
    rcl_std = np.std(rcl_list)
    pcn_mean = np.mean(pcn_list)
    pcn_std = np.std(pcn_list)
    time_mean = np.mean(time_list)
    time_std = np.std(time_list)
    print('AC %.4f %.4f ARI %.4f %.4f F1 %.4f %.4f '
          '(RCL %.4f %.4f PCN %.4f %.4f) TIME %.4f %.4f'
          % (acc_mean, acc_std, ARI_mean, ARI_std, F1_mean, F1_std,
             rcl_mean, rcl_std, pcn_mean, pcn_std, time_mean, time_std))

