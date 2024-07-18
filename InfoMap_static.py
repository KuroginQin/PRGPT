# Evaluate InfoMap on the static GP benchmark

import argparse
from infomap import Infomap
import time
import pickle
import random
from utils import *

def setup_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
gbl_rand_seed = 0

if __name__ == '__main__':
    # ====================
    setup_seed(gbl_rand_seed)

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
        im = Infomap(silent=True)
        for (src, dst) in edges:
            im.add_link(src, dst)
        clus_res = [-1 for _ in range(num_nodes)]

        # ====================
        time_s = time.time()
        im.run()
        lbl_set = set()
        for node in im.tree:
            if node.is_leaf:
                clus_res[node.node_id] = node.module_id - 1
                if node.module_id not in lbl_set:
                    lbl_set.add(node.module_id)
        time_e = time.time()
        inf_time = time_e - time_s

        # ====================
        num_clus_est = len(lbl_set)
        acc, ARI, recall, precision = quality_eva(gnd, clus_res)
        F1 = 2*recall*precision / (precision+recall)
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

