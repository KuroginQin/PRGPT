# Generate synthetic pre-training graphs using DC-SBM implemented by graph_tool

import graph_tool.all as gt
import numpy as np
import scipy.stats as stats
import pickle

# define discrete power law distribution
def discrete_power_law(a, min_v, max_v):
    x = np.arange(min_v, max_v + 1, dtype='float')
    pmf = x**a
    pmf /= pmf.sum()
    return stats.rv_discrete(values=(range(min_v, max_v+1), pmf))

# define the return function for in and out degrees
def degree_distribution_function(rv1, rv2):
    return (rv1.rvs(size=1), rv2.rvs(size=1))


# ====================
T = 1000 # Number of generated graphs
N_min = 2000 # Minimum number of nodes for a graph
N_max = 5000 # Maximum number of nodes for a graph
# ==========
density = 1 # sparsity parameter (1-density fraction of the edges will be removed)

# ====================
edge_list = []
gnd_list = []
for t in range(T):
    # ====================
    print('SNAP %d / %d' % (t+1, T))
    # ==========
    # Number of nodes
    N = np.random.randint(low=N_min, high=N_max+1)
    # ==========
    # Number of clusters/blocks
    num_blocks = np.random.randint(low=2, high=1000)
    # ==========
    # ratio between the total number of within-block edges and between-block edges
    ratio_within_over_between = np.random.uniform(low=2.5, high=5.01) # [2.5, 5]
    # block size heterogeneity - larger means the block sizes are more uneven
    block_size_heterogeneity = np.random.uniform(low=1, high=3.01) # # [1, 3]
    # parameters for the Power-Law degree distribution
    powerlaw_exponent = -np.random.uniform(low=2, high=3.51) # [2.0, 3.5]
    # ==========
    # set minimum & maximum node degree
    min_degree = round(min(5, N/(num_blocks*4))) # node degree range is adjusted lower when the blocks have few nodes
    max_degree = round(min(500, N/num_blocks))
    # set in-degree and out-degree distribution
    rv_indegree = discrete_power_law(powerlaw_exponent, min_degree, max_degree)
    rv_outdegree = discrete_power_law(powerlaw_exponent, min_degree, max_degree)

    block_distribution = np.random.dirichlet(np.ones(num_blocks)*10/block_size_heterogeneity, 1)[0]
    print('Block distribution: {}'.format(block_distribution))

    # draw block membership for each node
    block_membership_vector = np.where(np.random.multinomial(n=1, size=N, pvals=block_distribution))[1]
    true_partition = block_membership_vector


    # set the within-block and between-block edge strength accordingly
    def inter_block_strength(a, b):
        if a == b: # within block interaction strength
            return 1
        else:  # between block interaction strength
            avg_within_block_nodes = float(N) / num_blocks
            avg_between_block_nodes = N - avg_within_block_nodes
            return avg_within_block_nodes / avg_between_block_nodes / ratio_within_over_between

    # generate the graph
    if (float(gt.__version__[0:4]) >= 2.20): # specify inter-block strength through edge_probs in later versions
        g_sample, block_membership = gt.random_graph(N, lambda: degree_distribution_function(rv_indegree, rv_outdegree),
                                                     directed=True, model="blockmodel",
                                                     block_membership=block_membership_vector,
                                                     edge_probs=inter_block_strength, n_iter=10, verbose=False)
    else: # specify inter-block strength through vertex_corr in earlier versions
        g_sample, block_membership = gt.random_graph(N, lambda: degree_distribution_function(rv_indegree, rv_outdegree),
                                                     directed=True, model="blockmodel",
                                                     block_membership=block_membership_vector,
                                                     vertex_corr=inter_block_strength, n_iter=10, verbose=False)

    # remove (1-density) percent of the edges
    edge_filter = g_sample.new_edge_property('bool')
    edge_filter.a = stats.bernoulli.rvs(density, size=edge_filter.a.shape)
    g_sample.set_edge_filter(edge_filter)
    g_sample.purge_edges()

    # store the nodal block memberships in a vertex property
    g_sample.vertex_properties["block_membership"] = block_membership

    # compute and report basic statistics on the generated graph
    bg, bb, vcount, ecount, avp, aep = gt.condensation_graph(g_sample, block_membership, self_loops=True)
    edge_count_between_blocks = np.zeros((num_blocks, num_blocks))
    for e in bg.edges():
        edge_count_between_blocks[bg.vertex_index[e.source()], bg.vertex_index[e.target()]] = ecount[e]
    num_within_block_edges = sum(edge_count_between_blocks.diagonal())
    num_between_block_edges = g_sample.num_edges() - num_within_block_edges
    # print count statistics
    print('Number of nodes: {}'.format(N))
    print('Number of edges: {}'.format(g_sample.num_edges()))
    print('Avg. Number of nodes per block: {}'.format(N/num_blocks))
    print('# Within-block edges / # Between-blocks edges: {}'.format(num_within_block_edges/num_between_block_edges))

    # ====================
    edge_set = set()
    for e in g_sample.edges():
        # 0-base node indices
        src = int(e.source())
        dst = int(e.target())
        if src == dst: continue
        if src>dst:
            tmp = src
            src = dst
            dst = tmp
        if (src, dst) not in edge_set:
            edge_set.add((src, dst))
    edges = list(edge_set)
    edges = sorted(edges)
    # ==========
    #gnd = true_partition # 0-base label indices'
    gnd = []
    lbl_idx = 0
    lbl_map = {}
    for i in range(N):
        lbl = true_partition[i]
        if lbl not in lbl_map:
            gnd.append(lbl_idx)
            lbl_map[lbl] = lbl_idx
            lbl_idx += 1
        else:
            gnd.append(lbl_map[lbl])
    gnd = np.array(gnd)
    # ==========
    edge_list.append(edges)
    gnd_list.append(gnd)
    print()

# ====================
pkl_file = open('../data/ptn_edges_list.pickle', 'wb')
pickle.dump(edge_list, pkl_file)
pkl_file.close()
# ==========
pkl_file = open('../data/ptn_gnd_list.pickle', 'wb')
pickle.dump(gnd_list, pkl_file)
pkl_file.close()

