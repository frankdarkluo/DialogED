# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from torch.nn.functional import softplus
from torch.nn import Linear, Sequential, ReLU, Dropout

from hardkuma.position import get_relative_positions
from hardkuma.kuma import Kuma, HardKuma
#from entmax import sparsemax, entmax15, entmax_bisect
from gcn import GraphConvLayer, GatedGraphConvolution
#from model.structured import FineStructuredAttention
import numpy as np

MIN_CLAMP = 1e-3
MAX_CLAMP = 100

class KumaLayer(nn.Module):
    def __init__(self, dim, latent_type, sublayer_first, sublayer_second, gcn_dropout):
        super(KumaLayer, self).__init__()
        self.hidden_dim = dim
        self.latent_type = latent_type
        if self.latent_type == 'kuma_mtt':
#            lambda_p = 1
            self.gc1 = GatedGraphConvolution(self.hidden_dim, self.hidden_dim, lambda_p=1)
            self.gc2 = GatedGraphConvolution(self.hidden_dim, self.hidden_dim, lambda_p=1)
        elif self.latent_type == 'kuma':
            self.gc1 = GraphConvLayer(self.hidden_dim, gcn_dropout, sublayer_first)
            self.gc2 = GraphConvLayer(self.hidden_dim, gcn_dropout, sublayer_second)
        #self.fc = nn.Linear(2 * self.hidden_dim, opt.polarities_dim)

        self.max_relative_distance = 11

        self.rel_embedding = nn.Embedding(self.max_relative_distance * 2 + 1, 1)
        nn.init.xavier_normal_(self.rel_embedding.weight)

        self.kuma_attention = KumaSelfAttention(self.hidden_dim, self.hidden_dim, support=(-0.1, 1.1),
                                                dropout=0.2, dist_type='hardkuma', add_rel_dist=True,
                                                max_relative_distance=11, mask_diag=False,
                                                dist_embed=self.rel_embedding)
        #self.mtt_attention = FineStructuredAttention(self.hidden_dim)


    def forward(self,inputs, mask):

        if self.latent_type == 'kuma_mtt':
            kuma_adj = self.kuma_attention(inputs, inputs, inputs, mask)  # obtain latent graph
            mtt_adj = self.mtt_attention(inputs)

            x = self.gc1(inputs, mtt_adj, kuma_adj)  # do gcn

            weighted_x = x

            kuma_adj = self.kuma_attention(weighted_x, weighted_x, weighted_x, mask)  # obtain latent graph
            mtt_adj = self.mtt_attention(weighted_x)

            x = self.gc2(weighted_x, mtt_adj, kuma_adj)  # do gcn

        elif self.latent_type == 'kuma':
            kuma_adj = self.kuma_attention(inputs, inputs, inputs, mask)  # obtain latent graph
            x = self.gc1(kuma_adj,inputs)
            weighted_x = x
            kuma_adj = self.kuma_attention(weighted_x, weighted_x, weighted_x, mask)  # obtain latent graph
            x = self.gc2(kuma_adj,weighted_x)

        return x

class ScaledDotProductAttention(nn.Module):
    def __init__(self, d_k):
        super(ScaledDotProductAttention, self).__init__()
        self.d_k = d_k

    def forward(self, Q, K, V, attn_mask):
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(self.d_k)
        #print("[tlog] scores: " + str(scores.size()))
        #print("[tlog] scores: " + str(scores))
        #print("[tlog] attn_mask: " + str(attn_mask.size()))
        #print("[tlog] attn_mask: " + str(attn_mask))
        
        #scores = self._mask_padding(scores, attn_mask.unsqueeze(-1), -1e-9)
        #scores = self._mask_padding(scores, attn_mask.unsqueeze(-2), -1e-9)
        
        scores = self._mask_padding(scores, attn_mask.unsqueeze(-1), -1e9) #current the best for kuma 
        scores = self._mask_padding(scores, attn_mask.unsqueeze(-2), -1e9) 
        
        #scores = self._mask_padding(scores, attn_mask.unsqueeze(-1), 1e-9)
        #scores = self._mask_padding(scores, attn_mask.unsqueeze(-2), 1e-9)
        
        #scores = self._mask_padding(scores, attn_mask.unsqueeze(-1), -1e-25)
        #scores = self._mask_padding(scores, attn_mask.unsqueeze(-2), -1e-25)
        
        attn = nn.Softmax(dim=-1)(scores)
        #attn = entmax15(scores, dim=-1)
        context = torch.matmul(attn, V)
        return context, attn

    def _mask_padding(self, x, mask, value=0.):
        """
        Mask should be true/1 for valid positions, false/0 for invalid ones.
        :param x:
        :param mask:
        :return:
        """
        #print("[tlog] x: " + str(x))
        #print("[tlog] x.new: " + str(x.new_full([1], value)))
        #sys.exit(0)
        return torch.where(mask.byte(), x, x.new_full([1], value))


class KumaSelfAttention(nn.Module):
    def __init__(self, in_features, out_features, support=(-0.1, 1.1),
                 dropout=0.2, dist_type='hardkuma', add_rel_dist=True,
                 max_relative_distance=11, mask_diag=False, dist_embed=None):
        super(KumaSelfAttention, self).__init__()

        self.dist_type = dist_type
        self.activation = ReLU()
        self.dropout = Dropout(p=dropout)

        self.max_relative_distance = max_relative_distance
        self.mask_diag = mask_diag  # mask diagonal
        self.dist_embed = dist_embed
        self.add_rel_dist = add_rel_dist

        # For self attn
        self.d_model = in_features
        self.n_heads = 4 #4
        self.d_k = 80#80#64
        self.d_v = 80#80#64z
        self.a_W_Q = nn.Linear(self.d_model, self.d_k * self.n_heads)
        self.a_W_K = nn.Linear(self.d_model, self.d_k * self.n_heads)
        self.a_W_V = nn.Linear(self.d_model, self.d_k * self.n_heads)

        self.b_W_Q = nn.Linear(self.d_model, self.d_k * self.n_heads)
        self.b_W_K = nn.Linear(self.d_model, self.d_k * self.n_heads)
        self.b_W_V = nn.Linear(self.d_model, self.d_k * self.n_heads)

        self.a_attn = ScaledDotProductAttention(self.n_heads)
        self.b_attn = ScaledDotProductAttention(self.n_heads)

        self.a_score = nn.Linear(self.n_heads * self.d_v, self.n_heads * self.d_v)
        self.b_score = nn.Linear(self.n_heads * self.d_v, self.n_heads * self.d_v)

        self.layer_norm_a = nn.LayerNorm(self.n_heads * self.d_k)
        self.layer_norm_b = nn.LayerNorm(self.n_heads * self.d_k)

        self.support = support

        self.dist = None
        self.relu = nn.ReLU(inplace=True)

    def _mask_diagnoal(self, x, mask_value=0.):
        """block the diagonal so a word does not self-align"""
        eye = torch.eye(x.size(1), dtype=torch.uint8, device=x.device)
        return torch.where(eye, x.new_full([1], mask_value), x)

    def _add_rel_dists(self, x):
        """add matrix of relative distances"""
        rel_dists = get_relative_positions(
            x.size(1), self.max_relative_distance, device=x.device)
        rel_dists = self.dist_embed(rel_dists).squeeze(-1).unsqueeze(0)
        return x + rel_dists

    def forward(self, Q, K, V, mask):
        #print("[tlog] Q.size(): " + str(Q.size()))
        #print("[tlog] K.size(): " + str(K.size()))
        #print("[tlog] V.size(): " + str(V.size()))
        #print("[tlog] self.a_W_Q(Q).size(): " + str(self.a_W_Q(Q).size()))
        #print("[tlog] mask.size(): " + str(mask.size()))
        #print("[tlog] mask: " + str(mask))
        #sys.exit(0)
        batch_size = Q.size(0)

        a_q_s = self.a_W_Q(Q).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        a_k_s = self.a_W_Q(K).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2) # you can also try W_K and W_V
        a_v_s = self.a_W_Q(V).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)

        b_q_s = self.b_W_Q(Q).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        b_k_s = self.b_W_Q(K).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        b_v_s = self.b_W_Q(V).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        attn_mask = mask.unsqueeze(1).repeat(1, self.n_heads, 1)

        c_a, a_attn = self.a_attn(a_q_s, a_k_s, a_v_s, attn_mask)
        c_b, b_attn = self.b_attn(b_q_s, b_k_s, b_v_s, attn_mask)

        c_a = c_a.transpose(1, 2).contiguous().view(batch_size, -1,
                                                    self.n_heads * self.d_v)
        c_b = c_b.transpose(1, 2).contiguous().view(batch_size, -1,
                                                    self.n_heads * self.d_v)

        #tzy
        #if self.add_rel_dist:
        #    c_a = self._add_rel_dists(c_a)
        #    c_b = self._add_rel_dists(c_b)

        c_a = (self.a_score(c_a)) + c_a
        c_b = (self.b_score(c_b)) + c_b
        
        #c_a = self.relu(self.a_score(c_a)) + c_a
        #c_b = self.relu(self.b_score(c_b)) + c_b
        
        #c_a = self.a_score(self.relu(c_a)) + c_a
        #c_b = self.b_score(self.relu(c_b)) + c_b
        
        #c_a = (self.a_score(c_a).sigmoid() * self.a_score(c_a) ) + c_a
        #c_b = (self.b_score(c_b).sigmoid() * self.b_score(c_b) ) + c_b
        
        #c_a = self.relu(self.a_score(c_a)) + self.a_score(c_a) + c_a
        #c_b = self.relu(self.b_score(c_b)) + self.b_score(c_b) + c_b
        
        c_a = self.layer_norm_a(c_a)
        c_b = self.layer_norm_b(c_b)

        a = c_a @ c_a.transpose(-1, -2) 
        b = c_b @ c_b.transpose(-1, -2) 
        #print("[tlog] a.size: " + str(a.size()))
        #print("[tlog] b.size: " + str(b.size()))
        #sys.exit(0)
        # norm
        a = (a - a.mean()) / a.std()
        b = (b - b.mean()) / b.std()
        
        #a = nn.BatchNorm1d(a.size()[-1]).to(a.device)(a)
        #b = nn.BatchNorm1d(b.size()[-1]).to(b.device)(b)

        # add relative distances
        if self.add_rel_dist:
            a = self._add_rel_dists(a)
            b = self._add_rel_dists(b)
            
        #print("[tlog] a: " + str(a))
        #print("[tlog] b: " + str(b))
        
        a = softplus(a)
        b = softplus(b)

        a = a.clamp(MIN_CLAMP, MAX_CLAMP)

        b = b.clamp(MIN_CLAMP, MAX_CLAMP)

        # we return a distribution (from which we can sample if we want)
        if self.dist_type == "kuma":
            dist = Kuma([a, b])
        elif self.dist_type == "hardkuma":
            dist = HardKuma([a, b], support=self.support)
        else:
            raise ValueError("unknown dist")

        self.dist = dist

        if self.training:  # sample
            att = dist.sample() 
        else:  # predict deterministically
            p0 = dist.pdf(Q.new_zeros(()))
            p1 = dist.pdf(Q.new_ones(()))
            pc = 1. - p0 - p1  # prob. of sampling a continuous value [B, M]
            zero_one = torch.where(
                p0 > p1, Q.new_zeros([1]), Q.new_ones([1]))
            att = torch.where(pc < 0.5, zero_one, dist.mean())  # [B, M]
        
        #att = (att + a_attn.mean(dim=-1) + b_attn.mean(dim=-1))/3
        #print("[tlog] att: " + str(att))
        att = att * mask.unsqueeze(dim=-1).float()
        
        #att = att.ge(0.5).float() * att 
        
        #print("[tlog] att: " + str(att))
        #print("[tlog] att: " + str(att))
        #print("[tlog] att: " + str(att.size()))
        #print("[tlog] attn_mask: " + str(attn_mask.size()))
        #sys.exit(0)
        if self.mask_diag:
            att = self._mask_diagnoal(att, mask_value=0.)
            
        #att = (att - att.mean(dim=-1, keepdim=True)) / att.std(dim=-1, keepdim=True) #tzy
        
        return att  # [B, M]
