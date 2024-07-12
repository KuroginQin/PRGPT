import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.module import Module

class MDL_X1(Module):
    def __init__(self, emb_dim, num_feat_lyr, num_GNN_lyr, num_MLP_lyr_tmp, drop_rate):
        super(MDL_X1, self).__init__()
        # ====================
        self.emb_dim = emb_dim # Emb dim
        self.num_feat_lyr = num_feat_lyr # Number of MLP in feat red unit
        self.num_GNN_lyr = num_GNN_lyr # Number of GCN layers (i.e. step of message passing)
        self.num_MLP_lyr_tmp = num_MLP_lyr_tmp # Number of MLP layers for temp param
        self.drop_rate = drop_rate # Dropout rate
        # ==========
        # Feature reduction
        self.feat_red_lnrs = nn.ModuleList()
        self.feat_red_drops = nn.ModuleList()
        for l in range(self.num_feat_lyr):
            self.feat_red_lnrs.append(nn.Linear(in_features=self.emb_dim,
                                                out_features=self.emb_dim))
            self.feat_red_drops.append(nn.Dropout(p=self.drop_rate))
        # ==========
        # Emb linear mapping
        self.emb_lin_map = nn.Linear(in_features=self.emb_dim, out_features=self.emb_dim)
        # ==========
        # MLP for adaptive temp param
        self.lft_tmp_lnrs = nn.ModuleList()
        self.lft_tmp_drops = nn.ModuleList()
        self.rgt_tmp_lnrs = nn.ModuleList()
        self.rgt_tmp_drops = nn.ModuleList()
        for l in range(self.num_MLP_lyr_tmp):
            self.lft_tmp_lnrs.append(nn.Linear(in_features=self.emb_dim, out_features=self.emb_dim))
            self.lft_tmp_drops.append(nn.Dropout(p=self.drop_rate))
            self.rgt_tmp_lnrs.append(nn.Linear(in_features=self.emb_dim, out_features=self.emb_dim))
            self.rgt_tmp_drops.append(nn.Dropout(p=self.drop_rate))

    def forward(self, feat, sup):
        '''
        Rewrite the forward function
        :param feat: input feature
        :param sup: (sparse) GNN support
        :return: derived embedding
        '''
        # ====================
        feat_red = feat
        for l in range(self.num_feat_lyr):
            feat_red = self.feat_red_lnrs[l](feat_red)
            feat_red = self.feat_red_drops[l](feat_red)
            feat_red = torch.tanh(feat_red)
        # ====================
        # SGC
        GNN_in = feat
        GNN_out = None
        for l in range(self.num_GNN_lyr):
            GNN_out = torch.spmm(sup, GNN_in)
            GNN_in = GNN_out
        # ==========
        # Embedding linear map
        emb = self.emb_lin_map(GNN_out)
        emb = F.normalize(emb, dim=1, p=2)

        # ====================
        # Node-pair classifier w/ adaptive temp param
        lft_tmp_in = GNN_out
        rgt_tmp_in = GNN_out
        lft_tmp_out = None
        rgt_tmp_out = None
        for l in range(self.num_MLP_lyr_tmp):
            # ==========
            lft_tmp_out = self.lft_tmp_lnrs[l](lft_tmp_in)
            lft_tmp_out = self.lft_tmp_drops[l](lft_tmp_out)
            lft_tmp_out = torch.tanh(lft_tmp_out)
            if l < self.num_MLP_lyr_tmp - 1:
                lft_tmp_out = lft_tmp_out + lft_tmp_in  # Skip conn
            lft_tmp_in = lft_tmp_out
            # ===========
            rgt_tmp_out = self.rgt_tmp_lnrs[l](rgt_tmp_in)
            rgt_tmp_out = self.rgt_tmp_drops[l](rgt_tmp_out)
            rgt_tmp_out = torch.tanh(rgt_tmp_out)
            if l < self.num_MLP_lyr_tmp - 1:
                rgt_tmp_out = rgt_tmp_out + rgt_tmp_in  # Skip conn
            rgt_tmp_in = rgt_tmp_out

        return emb, lft_tmp_out, rgt_tmp_out


def get_edge_ind_est(emb_tnr, lft_tmp_tnr, rgt_tmp_tnr, src_idxs, dst_idxs):
    # ====================
    emb_src = emb_tnr[src_idxs, :]
    emb_dst = emb_tnr[dst_idxs, :]
    edge_ind_est = torch.sum(torch.multiply(emb_src, emb_dst), dim=1)
    # ==========
    tmp_src = lft_tmp_tnr[src_idxs, :]
    tmp_dst = rgt_tmp_tnr[dst_idxs, :]
    adp_tmp = torch.sum(torch.multiply(tmp_src, tmp_dst), dim=1)
    # ==========
    #edge_ind_est = torch.tanh(torch.mul(2*edge_ind_est - 2, adp_tmp)) + 1
    # =====
    edge_ind_est = torch.exp(torch.mul(2*edge_ind_est - 2, adp_tmp))

    return edge_ind_est

def get_node_pair_ind_est(emb_tnr, lft_tmp_tnr, rgt_tmp_tnr):
    # ====================
    node_pair_ind_est = torch.mm(emb_tnr, emb_tnr.t())
    adp_tmp = torch.mm(lft_tmp_tnr, rgt_tmp_tnr.t())
    # ==========
    #node_pair_ind_est = torch.tanh(torch.mul(2*node_pair_ind_est - 2, adp_tmp)) + 1
    # =====
    node_pair_ind_est = torch.exp(torch.mul(2*node_pair_ind_est - 2, adp_tmp))

    return node_pair_ind_est


def get_mod_max_loss(edge_ind_est, node_pair_ind_est, degs_tnr, num_edges, resolution=1.0):
    # ====================
    # First term
    fst = torch.sum(edge_ind_est)/num_edges
    # ==========
    # Second term
    lft_degs_tnr = torch.reshape(degs_tnr, (-1, 1))
    rgt_degs_tnr = torch.reshape(degs_tnr, (1, -1))
    scd = torch.multiply(lft_degs_tnr, torch.multiply(node_pair_ind_est, rgt_degs_tnr))
    scd = torch.sum(scd)/(4*num_edges*num_edges)
    # ==========
    # Modularity max
    loss = -(fst - resolution*scd)

    return loss

def get_BCE_loss(ind_gnd, ind_est, neg_param=1.0):
    # ====================
    # Binary cross-entropy
    pos_ind_est = torch.clamp(ind_est, min=1e-5)
    neg_ind_est = torch.clamp(1-ind_est, min=1e-5)
    loss = -(torch.multiply(ind_gnd, torch.log(pos_ind_est)) +
             neg_param*torch.multiply((1-ind_gnd), torch.log(neg_ind_est)))
    loss = torch.sum(loss)

    return loss

