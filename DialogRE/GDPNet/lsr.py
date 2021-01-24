import torch
import math
import torch.nn as nn
import torch.nn.functional as F
#from sparsemax.sparsemax import Sparsemax
from structure import StructuredAttention
from gcn import GraphConvLayer
#import utils.constant as constant
from structure import rnn_zero_state

PAD_ID = 0

class LSR(nn.Module):
    def __init__(self, hidden_size, dropout, num_layers, first_layer, second_layer):
        super(LSR, self).__init__()
        print("running LSR")

        self.mem_dim = hidden_size
        self.in_dim = hidden_size

        self.input_W_G = nn.Linear(self.in_dim, self.mem_dim)

        self.in_drop = nn.Dropout(dropout)
        self.num_layers = num_layers

        self.layers = nn.ModuleList()

        self.sublayer_first = first_layer #opt['sublayer_first']
        self.sublayer_second = second_layer #opt['sublayer_second']

        for i in range(self.num_layers):
            self.layers.append(StructuredAttention(self.mem_dim))
            if i == 0:
                self.layers.append(GraphConvLayer(self.mem_dim, dropout, self.sublayer_first))
            elif i == 1:
                self.layers.append(GraphConvLayer(self.mem_dim, dropout, self.sublayer_second))

        self.aggregate_W = nn.Linear(self.num_layers * self.mem_dim, self.mem_dim)


    def forward(self, vec):

        gcn_inputs = self.input_W_G(vec)

        layer_list = []
        outputs = gcn_inputs

        adj = None

        for i in range(len(self.layers)):
            if i % 2 == 0:
                adj = self.layers[i](outputs)
            else:
                outputs = self.layers[i](adj, outputs)
                layer_list.append(outputs)

        aggregate_out = torch.cat(layer_list, dim=2)
        dcgcn_output = self.aggregate_W(aggregate_out)
        #mask = (adj.sum(2) + adj.sum(1)).eq(0).unsqueeze(2)

        #return dcgcn_output, mask
        #return outputs, mask
        return dcgcn_output#, mask_flatten.unsqueeze(-1)
