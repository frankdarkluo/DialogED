import numpy as np
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter


class ChainCRF(nn.Module):
    def __init__(self, input_size, num_labels, bigram=True):
        '''

        Args:
            input_size: int
                the dimension of the input.
            num_labels: int
                the number of labels of the crf layer
            bigram: bool
                if apply bi-gram parameter.
        '''
        super(ChainCRF, self).__init__()
        self.input_size = input_size
        self.num_labels = num_labels + 1
        self.pad_label_id = num_labels
        self.bigram = bigram


        # state weight tensor
        self.state_net = nn.Linear(input_size, self.num_labels)
        if bigram:
            # transition weight tensor
            self.transition_net = nn.Linear(input_size, self.num_labels * self.num_labels)
            self.register_parameter('transition_matrix', None)
        else:
            self.transition_net = None
            self.transition_matrix = Parameter(torch.Tensor(self.num_labels, self.num_labels))

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.constant_(self.state_net.bias, 0.)
        if self.bigram:
            nn.init.xavier_uniform_(self.transition_net.weight)
            nn.init.constant_(self.transition_net.bias, 0.)
        else:
            nn.init.normal_(self.transition_matrix)

    def forward(self, input, mask=None):
        '''

        Args:
            input: Tensor
                the input tensor with shape = [batch, length, model_dim]
            mask: Tensor or None
                the mask tensor with shape = [batch, length]

        Returns: Tensor
            the energy tensor with shape = [batch, length, num_label, num_label]

        '''
        batch, length, _ = input.size()

        # compute out_s by tensor dot [batch, length, model_dim] * [model_dim, num_label]
        # thus out_s should be [batch, length, num_label] --> [batch, length, num_label, 1]
        out_s = self.state_net(input).unsqueeze(2)

        if self.bigram:
            # compute out_s by tensor dot: [batch, length, model_dim] * [model_dim, num_label * num_label]
            # the output should be [batch, length, num_label,  num_label]
            out_t = self.transition_net(input).view(batch, length, self.num_labels, self.num_labels)
            output = out_t + out_s
        else:
            # [batch, length, num_label, num_label]
            output = self.transition_matrix + out_s

        if mask is not None:
            output = output * mask.unsqueeze(2).unsqueeze(3)

        return output

    def loss(self, input, target, mask=None):
        '''

        Args:
            input: Tensor
                the input tensor with shape = [batch, length, model_dim]
            target: Tensor
                the tensor of target labels with shape [batch, length]
            mask:Tensor or None
                the mask tensor with shape = [batch, length]

        Returns: Tensor
                A 1D tensor for minus log likelihood loss [batch]
        '''
        batch, length, _ = input.size()
        energy = self(input, mask=mask)
        # shape = [length, batch, num_label, num_label]
        energy_transpose = energy.transpose(0, 1)
        # shape = [length, batch]
        target_transpose = target.transpose(0, 1)
        # shape = [length, batch, 1]
        mask_transpose = None
        if mask is not None:
            mask_transpose = mask.unsqueeze(2).transpose(0, 1)

        # shape = [batch, num_label]
        partition = None

        # shape = [batch]
        batch_index = torch.arange(0, batch).type_as(input).long()
        prev_label = input.new_full((batch, ), self.num_labels - 1).long()
        tgt_energy = input.new_zeros(batch)

        for t in range(length):
            # shape = [batch, num_label, num_label]
            curr_energy = energy_transpose[t]
            if t == 0:
                partition = curr_energy[:, -1, :]
            else:
                # shape = [batch, num_label]
                partition_new = torch.logsumexp(curr_energy + partition.unsqueeze(2), dim=1)
                if mask_transpose is None:
                    partition = partition_new
                else:
                    mask_t = mask_transpose[t]
                    partition = partition + (partition_new - partition) * mask_t
            tgt_energy += curr_energy[batch_index, prev_label, target_transpose[t]]
            prev_label = target_transpose[t]

        return torch.logsumexp(partition, dim=1) - tgt_energy

    def decode(self, input, mask=None, leading_symbolic=0):
        """

        Args:
            input: Tensor
                the input tensor with shape = [batch, length, model_dim]
            mask: Tensor or None
                the mask tensor with shape = [batch, length]
            leading_symbolic: nt
                number of symbolic labels leading in type alphabets (set it to 0 if you are not sure)

        Returns: Tensor
            decoding results in shape [batch, length]

        """

        energy = self(input, mask=mask)

        # Input should be provided as (n_batch, n_time_steps, num_labels, num_labels)
        # For convenience, we need to dimshuffle to (n_time_steps, n_batch, num_labels, num_labels)
        energy_transpose = energy.transpose(0, 1)

        # the last row and column is the tag for pad symbol. reduce these two dimensions by 1 to remove that.
        # also remove the first #symbolic rows and columns.
        # now the shape of energies_shuffled is [n_time_steps, b_batch, t, t] where t = num_labels - #symbolic - 1.
        energy_transpose = energy_transpose[:, :, leading_symbolic:-1, leading_symbolic:-1]

        length, batch_size, num_label, _ = energy_transpose.size()

        batch_index = torch.arange(0, batch_size).type_as(input).long()
        pi = input.new_zeros([length, batch_size, num_label])
        pointer = batch_index.new_zeros(length, batch_size, num_label)
        back_pointer = batch_index.new_zeros(length, batch_size)

        pi[0] = energy[:, 0, -1, leading_symbolic:-1]
        pointer[0] = -1
        for t in range(1, length):
            pi_prev = pi[t - 1]
            pi[t], pointer[t] = torch.max(energy_transpose[t] + pi_prev.unsqueeze(2), dim=1)

        _, back_pointer[-1] = torch.max(pi[-1], dim=1)
        for t in reversed(range(length - 1)):
            pointer_last = pointer[t + 1]
            back_pointer[t] = pointer_last[batch_index, back_pointer[t + 1]]

        return back_pointer.transpose(0, 1) + leading_symbolic