import numpy as np
import torch
from distributions import RectifiedStreched, BinaryConcrete
from gcn import GraphConvLayer
import torch.nn.functional as F
import torch.nn as nn

class MLPGate(torch.nn.Module):
    def __init__(self, input_size, hidden_size, bias=True):
        super().__init__()
        self.f = torch.nn.Sequential(
            torch.nn.utils.weight_norm(torch.nn.Linear(input_size, hidden_size)),
            torch.nn.Tanh(),
            torch.nn.utils.weight_norm(torch.nn.Linear(hidden_size, 1, bias=bias)),
        )
        if bias:
            self.f[-1].bias.data[:] = 5.0

    def forward(self, *args):
        return self.f(torch.cat(args, -1))


class MLPMaxGate(torch.nn.Module):
    def __init__(self, input_size, hidden_size, max_activation=10, bias=True):
        super().__init__()
        self.f = torch.nn.Sequential(
            torch.nn.utils.weight_norm(torch.nn.Linear(input_size, hidden_size)),
            torch.nn.Tanh(),
            #torch.nn.utils.weight_norm(torch.nn.Linear(hidden_size, hidden_size//2)),
            #torch.nn.Tanh(),
            torch.nn.utils.weight_norm(torch.nn.Linear(hidden_size, 1, bias=bias)),
            torch.nn.Tanh(),
        )
        self.bias = torch.nn.Parameter(torch.tensor(5.0))
        self.max_activation = max_activation

    def forward(self, *args):
        return self.f(torch.cat(args, -1)) * self.max_activation + self.bias


class GNNDiffMaskGate(torch.nn.Module):
    def __init__(
        self,
        hidden_size: int,
        hidden_attention: int,
        num_hidden_layers: int,
        max_position_embeddings: int,
        gcn_dropout: float,
        num_layer: int,
        sublayer_first: int,
        sublayer_second: int,
        l0_reg: bool = False,
        lasso_reg: bool = False,
        speaker_reg: bool = False,
        alpha: float = 1.,
        beta: float = 1.,
        gamma: float = 1.,
        gate_fn: torch.nn.Module = MLPMaxGate,
        gate_bias: bool = True,
        placeholder: bool = False,
        init_vector: torch.Tensor = None,
    ):
        super().__init__()

        self.g_hat = torch.nn.ModuleList(
            [
                gate_fn(hidden_size * 2, hidden_attention, bias=gate_bias)
                for _ in range(num_hidden_layers)
            ]
        )

        if placeholder:
            self.placeholder = torch.nn.Parameter(
                torch.nn.init.xavier_normal_(
                    torch.empty(1, max_position_embeddings, hidden_size,)
                )
                if init_vector is None
                else init_vector.view(1, 1, hidden_size).repeat(
                    1, max_position_embeddings, 1
                )
            )
        else:
            self.register_buffer(
                "placeholder", torch.zeros((1, 1, hidden_size,)),
            )

        self.layers = nn.ModuleList()
        self.num_layer = num_layer
        for i in range(self.num_layer):
            self.layers.append(GraphConvLayer(hidden_size, gcn_dropout, sublayer_first))
            self.layers.append(GraphConvLayer(hidden_size, gcn_dropout, sublayer_second))

        self.aggregate_W = nn.Linear(len(self.layers) * hidden_size, hidden_size)
        self.l0_reg = l0_reg
        self.speaker_reg = speaker_reg
        self.lasso_reg = lasso_reg
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def get_gates(self, hidden_states):
        bz = hidden_states.shape[0]
        max_seq_len = hidden_states.shape[1]
        logits_matrix = torch.zeros(bz, max_seq_len, max_seq_len).float().cuda()

        for r_index in range(max_seq_len):
                s = self.g_hat[0](hidden_states[:,r_index].view(bz,1,-1).expand(bz,max_seq_len,-1), hidden_states)
                logits_matrix[:,r_index] = s.squeeze(-1)

        #add sparsity regu
        logits_matrix = logits_matrix.unsqueeze(-1)
        dist = RectifiedStreched(
            BinaryConcrete(torch.full_like(logits_matrix, 0.2), logits_matrix), l=-0.2, r=1.0,
        )
        gates_full = dist.rsample().cumprod(-1)
        #expected_L0_full = dist.log_expected_L0().cumsum(-1)

        gates = gates_full[..., -1]
        #expected_L0 = expected_L0_full[..., -1]

        return  logits_matrix, dist, gates


    def get_losses(self, logits_matrix, dist, token_mask, speaker_mask = None):

        bz = logits_matrix.shape[0]
        max_seq_len = logits_matrix.shape[1]
        seq_lens = torch.sum(token_mask, dim=1).long()
        token_mask = token_mask.view(bz, max_seq_len, 1).expand(bz, max_seq_len, max_seq_len)
        # l0
        if self.l0_reg:
            expected_L0_full = dist.log_expected_L0().cumsum(-1)
            expected_L0 = expected_L0_full[..., -1]
            l0_loss = (expected_L0 * token_mask).sum(-1).sum(-1) / token_mask.sum(-1).sum(-1)
            l0_loss = torch.sum(l0_loss)
        else:
            l0_loss = 0

        if self.lasso_reg or self.speaker_reg:

            cdf0 = dist.cdf((torch.zeros(logits_matrix.size())).cuda())
            cdf0 = cdf0.squeeze(-1)
            pdf0 = torch.where(token_mask, cdf0, cdf0.new_zeros([1]))

        #speaker
        if self.speaker_reg:
            speaker_loss = torch.sum(pdf0 * speaker_mask) / torch.sum(speaker_mask)
        else:
            speaker_loss = 0

        #lasso
        if self.lasso_reg:
            pdf_nonzero = 1. - pdf0  # [B, T] # all non-zero
            pdf_nonzero = torch.where(token_mask, pdf_nonzero, pdf_nonzero.new_zeros([1]))

            lasso_cost_list = []
            for i in range(bz):
                zt_zero_i = pdf0[:, i, :][:,:-1]
                ztp1_nonzero_i = pdf_nonzero[:, i, :][:,1:]

                zt_nonzero_i = pdf_nonzero[:, i, :][:,:-1]
                ztp1_zero_i = pdf0[:, i, :][:,1:]

                # number of transitions per sentence normalized by length
                lasso_cost = zt_zero_i * ztp1_nonzero_i + zt_nonzero_i * ztp1_zero_i
                lasso_cost = lasso_cost * token_mask[:, i, :].float()[:, :-1]
                lasso_cost = lasso_cost.sum(1) / (seq_lens + 1e-9)  # [B]
                lasso_cost = lasso_cost.sum() / bz
                lasso_cost_list.append(lasso_cost)

            lasso_loss = torch.sum(torch.stack(lasso_cost_list))
        else:
            lasso_loss = 0

        return  l0_loss, speaker_loss, lasso_loss

    def forward(self, hidden_states, token_mask, speaker_mask = None):
        x = hidden_states
        layer_list = []
        logits_matrix, dist, gates = None, None, None

        for i in range(len(self.layers)):
            logits_matrix, dist, gates = self.get_gates(x)
            x = self.layers[i](gates, hidden_states)
            layer_list.append(x)

        l0_loss, speaker_reg, lasso_reg = self.get_losses(logits_matrix, dist,token_mask,speaker_mask)

        # gates, expected_l0 = self.get_gates_expectedl0(x,mask)
        # output = self.gc2(gates, hidden_states)

        aggregate_out = torch.cat(layer_list, dim=-1)
        output = self.aggregate_W(aggregate_out)

        l0_loss = self.alpha * l0_loss + self.beta * speaker_reg + self.gamma * lasso_reg

        return output, l0_loss

class DiffMaskGateHidden(torch.nn.Module):
    def __init__(
        self,
        hidden_size: int,
        hidden_attention: int,
        num_hidden_layers: int,
        max_position_embeddings: int,
        gate_fn: torch.nn.Module = MLPMaxGate,
        gate_bias: bool = True,
        placeholder: bool = False,
        init_vector: torch.Tensor = None,
    ):
        super().__init__()

        self.g_hat = torch.nn.ModuleList(
            [
                gate_fn(hidden_size, hidden_attention, bias=gate_bias)
                for _ in range(num_hidden_layers)
            ]
        )

        if placeholder:
            self.placeholder = torch.nn.ParameterList(
                [
                    torch.nn.Parameter(
                        torch.nn.init.xavier_normal_(
                            torch.empty(1, max_position_embeddings, hidden_size,)
                        )
                        if init_vector is None
                        else init_vector.view(1, 1, hidden_size).repeat(
                            1, max_position_embeddings, 1
                        )
                    )
                    for _ in range(num_hidden_layers)
                ]
            )
        else:
            self.register_buffer(
                "placeholder", torch.zeros((num_hidden_layers, 1, 1, hidden_size,)),
            )

    def forward(self, hidden_states, mask, layer_pred):

        if layer_pred is not None:
            logits = self.g_hat[layer_pred](hidden_states[layer_pred])
        else:
            logits = torch.cat(
                [self.g_hat[i](hidden_states[i]) for i in range(len(hidden_states))], -1
            )

        dist = RectifiedStreched(
            BinaryConcrete(torch.full_like(logits, 0.2), logits), l=-0.2, r=1.0,
        )

        gates_full = dist.rsample()
        expected_L0_full = dist.log_expected_L0()

        gates = gates_full if layer_pred is not None else gates_full[..., :1]
        expected_L0 = (
            expected_L0_full if layer_pred is not None else expected_L0_full[..., :1]
        )

        return (
            hidden_states[layer_pred if layer_pred is not None else 0] * gates
            + self.placeholder[layer_pred if layer_pred is not None else 0][
                :,
                : hidden_states[layer_pred if layer_pred is not None else 0].shape[-2],
            ]
            * (1 - gates),
            gates.squeeze(-1),
            expected_L0.squeeze(-1),
            gates_full,
            expected_L0_full,
        )


class PerSampleGate(torch.nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_hidden_layers: int,
        max_position_embeddings: int,
        batch_size: int = 1,
        placeholder: bool = False,
        init_vector: torch.Tensor = None,
    ):
        super().__init__()

        self.logits = torch.nn.Parameter(
            torch.full((batch_size, max_position_embeddings, num_hidden_layers), 5.0)
        )

        if placeholder:
            self.placeholder = torch.nn.Parameter(
                torch.nn.init.xavier_normal_(
                    torch.empty(
                        batch_size,
                        num_hidden_layers,
                        max_position_embeddings,
                        hidden_size,
                    )
                )
                if init_vector is None
                else init_vector.view(1, num_hidden_layers, 1, hidden_size).repeat(
                    batch_size, 1, max_position_embeddings, 1
                )
            )
        else:
            self.register_buffer(
                "placeholder", torch.zeros((1, num_hidden_layers, 1, hidden_size))
            )


class PerSampleDiffMaskGate(PerSampleGate):
    def forward(self, hidden_states, mask, layer_pred):

        dist = RectifiedStreched(
            BinaryConcrete(torch.full_like(self.logits, 0.2), self.logits),
            l=-0.2,
            r=1.0,
        )

        gates_full = dist.rsample()
        expected_L0_full = dist.log_expected_L0()

        gates = gates_full[..., layer_pred]
        expected_L0 = expected_L0_full[..., layer_pred]

        return (
            hidden_states[layer_pred] * gates.unsqueeze(-1)
            + self.placeholder[:, layer_pred, : hidden_states[layer_pred].shape[-2]]
            * (1 - gates).unsqueeze(-1),
            gates,
            expected_L0,
            gates_full,
            expected_L0_full,
        )

class PerSampleREINFORCEGate(PerSampleGate):
    def forward(self, hidden_states, mask, layer_pred):

        dist = torch.distributions.Bernoulli(logits=self.logits)

        gates_full = dist.sample()
        expected_L0_full = dist.log_prob(1.0)

        gates = gates_full[..., layer_pred]
        expected_L0 = expected_L0_full[..., layer_pred]

        return (
            hidden_states[layer_pred] * gates.unsqueeze(-1)
            + self.placeholder[:, layer_pred, : hidden_states[layer_pred].shape[-2]]
            * (1 - gates).unsqueeze(-1),
            gates,
            expected_L0,
            gates_full,
            expected_L0_full,
        )
