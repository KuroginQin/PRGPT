# Evaluate InfoMap on the streaming GP benchmark

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
    parser.add_argument('--ind', type=int, default=1)
    args = parser.parse_args()
    # ==========
    num_nodes = args.N # Number of nodes
    snap_idx = args.ind # Index of graph to be partitioned, 1-5
    num_step = 10 # Number of streaming steps
    num_nodes_per_step = int(num_nodes / num_step)
    mu = 2.5 # ratio_within_over_between
    beta = 3 # block_size_heterogeneity

    # ===================
    # Load the dataset
    edges_list_file = open('data/stream_%d_%d_%.1f_%.1f_edges_list.pickle'
                           % (snap_idx, num_nodes, mu, beta), 'rb')
    edges_list = pickle.load(edges_list_file)
    edges_list_file.close()
    # ==========
    gnd_file = open('data/stream_%d_%d_%.1f_%.1f_gnd.pickle'
                    % (snap_idx, num_nodes, mu, beta), 'rb')
    gnd = pickle.load(gnd_file)
    gnd_file.close()
    # ==========
    if int(np.min(gnd)) == 0:
        num_clus_gnd = int(np.max(gnd)) + 1 # Number of clusters
    else:
        num_clus_gnd = int(np.max(gnd))
        gnd = [lbl - 1 for lbl in gnd]
    # ==========
    if np.min(edges_list[0]) == 1:
        zero_base = False
    else:
        zero_base = True

    # ====================
    acc_edges = [] # Accumulative edge list
    end_idx = num_nodes_per_step
    for t in range(num_step):
        # ==========
        print('STEP %d / %d' % (t+1, num_step))
        edges = edges_list[t]
        for (src, dst) in edges:
            if not zero_base: # Ensure that node indices are w/ 0-base
                acc_edges.append((src-1, dst-1))
                acc_edges.append((dst-1, src-1))
            else:
                acc_edges.append((src, dst))
                acc_edges.append((dst, src))
        # ==========
        crt_gnd = gnd[0:end_idx] # Ground-truth w.r.t. current (accumulative) topo

        # ====================
        im = Infomap(silent=True)
        for (src, dst) in acc_edges:
            im.add_link(src, dst)
        clus_res = [-1 for _ in range(end_idx)]

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
        acc, ARI, recall, precision = quality_eva(crt_gnd, clus_res)
        F1 = 2*recall*precision / (precision+recall)
        print('Gnd-K %d Alg-K %d AC %.4f ARI %.4f F1 %.4f (RCL %.4f PCN %.4f) TIME %f'
              % (num_clus_gnd, num_clus_est, acc, ARI,
                 F1, recall, precision, inf_time))

        # ====================
        end_idx += num_nodes_per_step

