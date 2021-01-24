import torch
import math
import torch.nn as nn
import torch.nn.functional as F
#from utils import constant, torch_utils
from torch.autograd import Variable
INFINITY_NUMBER = 1e12

class StructuredAttention(nn.Module):
    def __init__(self, sent_hiddent_size):#, py_version):
        super(StructuredAttention, self).__init__()

        self.str_dim_size = sent_hiddent_size #- self.sem_dim_size

        self.model_dim = sent_hiddent_size

        self.linear_keys = nn.Linear(self.model_dim, self.model_dim)
        self.linear_query = nn.Linear(self.model_dim, self.model_dim)
        self.linear_root = nn.Linear(self.model_dim, 1)

    def forward(self, input):  # batch*sent * token * hidden
        batch_size, token_size, dim_size = input.size()

        key = self.linear_keys(input)
        query = self.linear_query(input)
        f_i = self.linear_root(input).squeeze(-1)
        query = query / math.sqrt(self.model_dim)
        f_ij = torch.matmul(query, key.transpose(1, 2))

        mask = torch.ones(f_ij.size(1), f_ij.size(1)) - torch.eye(f_ij.size(1), f_ij.size(1))
        mask = mask.unsqueeze(0).expand(f_ij.size(0), mask.size(0), mask.size(1)).cuda()
        A_ij = torch.exp(f_ij) * mask

        tmp = torch.sum(A_ij, dim=1)  # nan: dimension
        res = torch.zeros(batch_size, token_size, token_size).cuda()

        res.as_strided(tmp.size(), [res.stride(0), res.size(2) + 1]).copy_(tmp)
        L_ij = -A_ij + res  # A_ij has 0s as diagonals

        L_ij_bar = L_ij
        L_ij_bar[:, 0, :] = f_i

        LLinv = torch.inverse(L_ij_bar)

        d0 = f_i * LLinv[:, :, 0]

        LLinv_diag = torch.diagonal(LLinv, dim1=-2, dim2=-1).unsqueeze(2)

        tmp1 = (A_ij.transpose(1, 2) * LLinv_diag).transpose(1, 2)
        tmp2 = A_ij * LLinv.transpose(1, 2)

        temp11 = torch.zeros(batch_size, token_size, 1)
        temp21 = torch.zeros(batch_size, 1, token_size)

        temp12 = torch.ones(batch_size, token_size, token_size - 1)
        temp22 = torch.ones(batch_size, token_size - 1, token_size)

        mask1 = torch.cat([temp11, temp12], 2).cuda()
        mask2 = torch.cat([temp21, temp22], 1).cuda()

        dx = mask1 * tmp1 - mask2 * tmp2

        d = torch.cat([d0.unsqueeze(1), dx], dim=1)
        df = d.transpose(1, 2)

        att = df[:, :, 1:]

        return att


def pool(h, mask, type='max'):
    mask = mask[:,:h.shape[1]]
    if type == 'max':
        if h.shape[0] != mask.shape[0] or h.shape[1] != mask.shape[1]:
            mask = mask[:, : h.shape[1]]
            print("error")
            print('\n')
        #if h.shape[0]
        #if h.shape[1] != mask.shape[1]:
        #    print("error")
        h = h.masked_fill(mask, -INFINITY_NUMBER)
        return torch.max(h, 1)[0]
    elif type == 'avg':
        if h.shape[0] != mask.shape[0] or h.shape[1] != mask.shape[1]:
            mask = mask[:, : h.shape[1]]
            print("error")
            print('\n')
        h = h.masked_fill(mask, 0)
        return h.sum(1) / (mask.size(1) - mask.float().sum(1))
    else:
        h = h.masked_fill(mask, 0)
        return h.sum(1)


def rnn_zero_state(batch_size, hidden_dim, num_layers, bidirectional=True):
    total_layers = num_layers * 2 if bidirectional else num_layers
    state_shape = (total_layers, batch_size, hidden_dim)
    h0 = c0 = Variable(torch.zeros(*state_shape), requires_grad=False)
    return h0.cuda(), c0.cuda()

class SelfAttention(nn.Module):
    def __init__(self, input_size, dropout):
        super().__init__()
        # self.dropout = LockedDropout(dropout)
        self.input_linear = nn.Linear(input_size, 1, bias=False)
        self.dot_scale = nn.Parameter(torch.Tensor(input_size).uniform_(1.0 / (input_size ** 0.5)))
        self.output_linear = nn.Linear(2 * input_size, input_size, bias=False)

    def forward(self, input, memory, mask):

        input_dot = self.input_linear(input) #nan: cal the weight for the same word
        cross_dot = torch.bmm(input * self.dot_scale, memory.permute(0, 2, 1).contiguous())
        att = input_dot + cross_dot
        att = att - 1e30 * mask[:,None]

        weight_one = F.softmax(att, dim=-1)
        output_one = torch.bmm(weight_one, memory)

        return output_one #self.output_linear(torch.cat([input, output_one], dim=-1))


def _getMatrixTree_multi(scores, root):
    A = scores.exp()
    R = root.exp()

    L = torch.sum(A, 1)
    L = torch.diag_embed(L)
    L = L - A
    LL = L + torch.diag_embed(R)
    LL_inv = torch.inverse(LL)  # batch_l, doc_l, doc_l
    LL_inv_diag = torch.diagonal(LL_inv, 0, 1, 2)
    d0 = R * LL_inv_diag
    LL_inv_diag = torch.unsqueeze(LL_inv_diag, 2)

    _A = torch.transpose(A, 1, 2)
    _A = _A * LL_inv_diag
    tmp1 = torch.transpose(_A, 1, 2)
    tmp2 = A * torch.transpose(LL_inv, 1, 2)

    d = tmp1 - tmp2
    return d, d0


class Inter_StructuredAttention(nn.Module):
    def __init__(self, model_dim, dropout=0.1):
        self.model_dim = model_dim

        super(Inter_StructuredAttention, self).__init__()

        self.linear_keys = nn.Linear(model_dim, self.model_dim)
        self.linear_query = nn.Linear(model_dim, self.model_dim)
        self.linear_root = nn.Linear(model_dim, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):


        key = self.linear_keys(x)
        query = self.linear_query(x)
        root = self.linear_root(x).squeeze(-1)

        query = query / math.sqrt(self.model_dim)
        scores = torch.matmul(query, key.transpose(1, 2))

        mask = mask.float()
        root = root - mask.squeeze(1) * 50
        root = torch.clamp(root, min=-40)
        scores = scores - mask * 50
        scores = scores - torch.transpose(mask, 1, 2) * 50
        scores = torch.clamp(scores, min=-40)
        # _logits = _logits + (tf.transpose(bias, [0, 2, 1]) - 1) * 40
        # _logits = tf.clip_by_value(_logits, -40, 1e10)

        d, d0 = _getMatrixTree_multi(scores, root)
        attn = torch.transpose(d, 1,2)
        if mask is not None:
            mask = mask.expand_as(scores).bool()
            attn = attn.masked_fill(mask, 0)

        return attn, d0