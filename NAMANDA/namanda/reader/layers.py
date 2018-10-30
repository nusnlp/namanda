#!/usr/bin/env python3
"""Definitions of model layers/NN modules"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


# ------------------------------------------------------------------------------
# Modules
# ------------------------------------------------------------------------------


class StackedBRNN(nn.Module):
    """Stacked Bi-directional RNNs. (Taken from DrQA repo)

    Differs from standard PyTorch library in that it has the option to save
    and concat the hidden states between layers. (i.e. the output hidden size
    for each sequence input is num_layers * hidden_size).
    """

    def __init__(self, input_size, hidden_size, num_layers,
                 dropout_rate=0, dropout_output=False, rnn_type=nn.LSTM,
                 concat_layers=False, padding=False):
        super(StackedBRNN, self).__init__()
        self.padding = padding
        self.dropout_output = dropout_output
        self.dropout_rate = dropout_rate
        self.num_layers = num_layers
        self.concat_layers = concat_layers
        self.rnns = nn.ModuleList()
        for i in range(num_layers):
            input_size = input_size if i == 0 else 2 * hidden_size
            self.rnns.append(rnn_type(input_size, hidden_size,
                                      num_layers=1,
                                      bidirectional=True))

    def forward(self, x, x_mask):
        """Encode either padded or non-padded sequences.

        Can choose to either handle or ignore variable length sequences.
        Always handle padding in eval.

        Args:
            x: batch * len * hdim
            x_mask: batch * len (1 for padding, 0 for true)
        Output:
            x_encoded: batch * len * hdim_encoded
        """
        if x_mask.data.sum() == 0:
            # No padding necessary.
            output = self._forward_unpadded(x, x_mask)
        elif self.padding or not self.training:
            # Pad if we care or if its during eval.
            output = self._forward_padded(x, x_mask)
        else:
            # We don't care.
            output = self._forward_unpadded(x, x_mask)

        return output.contiguous()

    def _forward_unpadded(self, x, x_mask):
        """Faster encoding that ignores any padding."""
        # Transpose batch and sequence dims
        x = x.transpose(0, 1)

        # Encode all layers
        outputs = [x]
        for i in range(self.num_layers):
            rnn_input = outputs[-1]

            # Apply dropout to hidden input
            if self.dropout_rate > 0:
                rnn_input = F.dropout(rnn_input,
                                      p=self.dropout_rate,
                                      training=self.training)
            # Forward
            rnn_output = self.rnns[i](rnn_input)[0]
            outputs.append(rnn_output)

        # Concat hidden layers
        if self.concat_layers:
            output = torch.cat(outputs[1:], 2)
        else:
            output = outputs[-1]

        # Transpose back
        output = output.transpose(0, 1)

        # Dropout on output layer
        if self.dropout_output and self.dropout_rate > 0:
            output = F.dropout(output,
                               p=self.dropout_rate,
                               training=self.training)
        return output

    def _forward_padded(self, x, x_mask):
        """Slower (significantly), but more precise, encoding that handles
        padding.
        """
        # Compute sorted sequence lengths
        lengths = x_mask.data.eq(0).long().sum(1).squeeze()
        _, idx_sort = torch.sort(lengths, dim=0, descending=True)
        _, idx_unsort = torch.sort(idx_sort, dim=0)

        lengths = list(lengths[idx_sort])
        idx_sort = Variable(idx_sort)
        idx_unsort = Variable(idx_unsort)

        # Sort x
        x = x.index_select(0, idx_sort)

        # Transpose batch and sequence dims
        x = x.transpose(0, 1)

        # Pack it up
        rnn_input = nn.utils.rnn.pack_padded_sequence(x, lengths)

        # Encode all layers
        outputs = [rnn_input]
        for i in range(self.num_layers):
            rnn_input = outputs[-1]

            # Apply dropout to input
            if self.dropout_rate > 0:
                dropout_input = F.dropout(rnn_input.data,
                                          p=self.dropout_rate,
                                          training=self.training)
                rnn_input = nn.utils.rnn.PackedSequence(dropout_input,
                                                        rnn_input.batch_sizes)
            outputs.append(self.rnns[i](rnn_input)[0])

        # Unpack everything
        for i, o in enumerate(outputs[1:], 1):
            outputs[i] = nn.utils.rnn.pad_packed_sequence(o)[0]

        # Concat hidden layers or take final
        if self.concat_layers:
            output = torch.cat(outputs[1:], 2)
        else:
            output = outputs[-1]

        # Transpose and unsort
        output = output.transpose(0, 1)
        output = output.index_select(0, idx_unsort)

        # Pad up to original batch sequence length
        if output.size(1) != x_mask.size(1):
            padding = torch.zeros(output.size(0),
                                  x_mask.size(1) - output.size(1),
                                  output.size(2)).type(output.data.type())
            output = torch.cat([output, Variable(padding)], 1)

        # Dropout on output layer
        if self.dropout_output and self.dropout_rate > 0:
            output = F.dropout(output,
                               p=self.dropout_rate,
                               training=self.training)
        return output


class SeqAttnMatch(nn.Module):
    """Given sequences X and Y, match sequence Y to each element in X.

    * o_i = sum(alpha_j * y_j) for i in X
    * alpha_j = softmax(y_j * x_i)
    """
    def __init__(self, input_size, identity=False):
        super(SeqAttnMatch, self).__init__()
        if not identity:
            self.linear = nn.Linear(input_size, input_size)
        else:
            self.linear = None

    def forward(self, x, y, y_mask):
        """
        Args:
            x: batch * len1 * hdim
            y: batch * len2 * hdim
            y_mask: batch * len2 (1 for padding, 0 for true)
        Output:
            matched_seq: batch * len1 * hdim
        """
        # Project vectors
        if self.linear:
            x_proj = self.linear(x.view(-1, x.size(2))).view(x.size())
            x_proj = F.relu(x_proj)
            y_proj = self.linear(y.view(-1, y.size(2))).view(y.size())
            y_proj = F.relu(y_proj)
        else:
            x_proj = x
            y_proj = y

        # Compute scores
        scores = x_proj.bmm(y_proj.transpose(2, 1))

        # Mask padding
        y_mask = y_mask.unsqueeze(1).expand(scores.size())
        scores.data.masked_fill_(y_mask.data, -float('inf'))

        # Normalize with softmax
        alpha_flat = F.softmax(scores.view(-1, y.size(1)))
        alpha = alpha_flat.view(-1, x.size(1), y.size(1))

        # Take weighted average
        matched_seq = alpha.bmm(y)
        return matched_seq


class BilinearSeqAttn(nn.Module):
    """A bilinear attention layer over a sequence X w.r.t y:

    * o_i = softmax(x_i'Wy) for x_i in X.

    Optionally don't normalize output weights.
    """

    def __init__(self, x_size, y_size, identity=False, normalize=True):
        super(BilinearSeqAttn, self).__init__()
        self.normalize = normalize

        # If identity is true, we just use a dot product without transformation.
        if not identity:
            self.linear = nn.Linear(y_size, x_size)
        else:
            self.linear = None

    def forward(self, x, y, x_mask):
        """
        Args:
            x: batch * len * hdim1
            y: batch * hdim2
            x_mask: batch * len (1 for padding, 0 for true)
        Output:
            alpha = batch * len
        """
        Wy = self.linear(y) if self.linear is not None else y
        xWy = x.bmm(Wy.unsqueeze(2)).squeeze(2)
        xWy.data.masked_fill_(x_mask.data, -float('inf'))
        if self.normalize:
            if self.training:
                # In training we output log-softmax for NLL
                alpha = F.log_softmax(xWy)
            else:
                # ...Otherwise 0-1 probabilities
                alpha = F.softmax(xWy)
        else:
            alpha = xWy.exp()
        return alpha


class LinearSeqAttn(nn.Module):
    """Self attention over a sequence:

    * o_i = softmax(Wx_i) for x_i in X.
    """

    def __init__(self, input_size):
        super(LinearSeqAttn, self).__init__()
        self.linear = nn.Linear(input_size, 1)

    def forward(self, x, x_mask):
        """
        Args:
            x: batch * len * hdim
            x_mask: batch * len (1 for padding, 0 for true)
        Output:
            alpha: batch * len
        """
        x_flat = x.view(-1, x.size(-1))
        scores = self.linear(x_flat).view(x.size(0), x.size(1))
        scores.data.masked_fill_(x_mask.data, -float('inf'))
        alpha = F.softmax(scores)
        return alpha


class BilinearSeqAttnwithNIL(nn.Module):
    """A bilinear attention layer over a sequence X w.r.t y:

    * o_i = softmax(x_i'Wy) for x_i in X.

    Optionally don't normalize output weights.
    """

    def __init__(self, x_size, y_size, identity=False, normalize=True):
        super(BilinearSeqAttnwithNIL, self).__init__()
        self.normalize = normalize

        # If identity is true, we just use a dot product without transformation.
        if not identity:
            self.linear = nn.Linear(y_size, x_size)
        else:
            self.linear = None

        self.nil_linear = nn.Linear(x_size, 1)

    def forward(self, x, y, x_mask):
        """
        Args:
            x: batch * len * hdim1
            y: batch * hdim2
            x_mask: batch * len (1 for padding, 0 for true)
        Output:
            alpha = batch * len
        """
        x_nil_rep = x.narrow(1,0,1)
        nil_score = self.nil_linear(x_nil_rep)
        nil_score = nil_score.view(x_nil_rep.size(0),1)

        x = x.narrow(1, 1, x.size(1)-1)

        Wy = self.linear(y) if self.linear is not None else y
        xWy = x.bmm(Wy.unsqueeze(2)).squeeze(2)
        xWy = torch.cat([nil_score, xWy], 1)
        xWy.data.masked_fill_(x_mask.data, -float('inf'))
        if self.normalize:
            if self.training:
                # In training we output log-softmax for NLL
                alpha = F.log_softmax(xWy)
            else:
                # ...Otherwise 0-1 probabilities
                alpha = F.softmax(xWy)
        else:
            alpha = xWy.exp()
        return alpha


class SeqAttnMat(nn.Module):
    """Given sequences X and Y, calculate the attention matrix.

    * o_i = sum(alpha_j * y_j) for i in X
    * alpha_j = softmax(y_j * x_i)
    """

    def __init__(self, input_size, identity=False):
        super(SeqAttnMat, self).__init__()
        if not identity:
            self.linear = nn.Linear(input_size, input_size)
        else:
            self.linear = None

    def forward(self, x, y, y_mask):
        """
        Args:
            x: batch * len1 * hdim
            y: batch * len2 * hdim
            y_mask: batch * len2 (1 for padding, 0 for true)
        Output:
            attn_scores: batch * len1 * len2
        """
        # Project vectors
        if self.linear:
            x_proj = self.linear(x.view(-1, x.size(2))).view(x.size())
            x_proj = F.relu(x_proj)
            y_proj = self.linear(y.view(-1, y.size(2))).view(y.size())
            y_proj = F.relu(y_proj)
        else:
            x_proj = x
            y_proj = y

        # Compute scores
        scores = x_proj.bmm(y_proj.transpose(2, 1))

        # Mask padding
        y_mask = y_mask.unsqueeze(1).expand(scores.size())
        scores.data.masked_fill_(y_mask.data, -float('inf'))

        # Normalize with softmax
        alpha_flat = F.softmax(scores.view(-1, y.size(1)))
        alpha = alpha_flat.view(-1, x.size(1), y.size(1))

        return scores, alpha


class SeqAttnMatwithSMin(nn.Module):
    """Given sequences X and Y, calculate the attention matrix.

    * o_i = sum(alpha_j * y_j) for i in X
    * alpha_j = softmax(y_j * x_i)
    """

    def __init__(self, input_size, identity=False):
        super(SeqAttnMatwithSMin, self).__init__()
        if not identity:
            self.linear = nn.Linear(input_size, input_size)
        else:
            self.linear = None

    def forward(self, x, y, y_mask):
        """
        Args:
            x: batch * len1 * hdim
            y: batch * len2 * hdim
            y_mask: batch * len2 (1 for padding, 0 for true)
        Output:
            attn_scores: batch * len1 * len2
        """
        # Project vectors
        if self.linear:
            x_proj = self.linear(x.view(-1, x.size(2))).view(x.size())
            x_proj = F.relu(x_proj)
            y_proj = self.linear(y.view(-1, y.size(2))).view(y.size())
            y_proj = F.relu(y_proj)
        else:
            x_proj = x
            y_proj = y

        # Compute scores
        scores = x_proj.bmm(y_proj.transpose(2, 1))
        scores_nil = x_proj.bmm(y_proj.transpose(2, 1))

        # Mask padding
        y_mask = y_mask.unsqueeze(1).expand(scores.size())
        scores.data.masked_fill_(y_mask.data, -float('inf'))
        scores_nil.data.masked_fill_(y_mask.data, float('inf'))

        # Normalize with softmax
        alpha_flat = F.softmax(scores.view(-1, y.size(1)))
        alpha = alpha_flat.view(-1, x.size(1), y.size(1))

        alpha_flat_min = F.softmin(scores_nil.view(-1, y.size(1)))
        alpha_min = alpha_flat_min.view(-1, x.size(1), y.size(1))

        return scores, alpha, alpha_min


class GatedEncoding(nn.Module):
    """Gating over a sequence:

    * o_i = sigmoid(Wx_i) * x_i for x_i in X.
    """

    def __init__(self, input_size):
        super(GatedEncoding, self).__init__()
        self.linear = nn.Linear(input_size, input_size)

    def forward(self, x):
        """
        Args:
            x: batch * len * hdim
        Output:
            gated_x: batch * len * hdim
        """
        gate = self.linear(x.view(-1, x.size(2))).view(x.size())
        gate = F.sigmoid(gate)
        gated_x = torch.mul(gate, x)


        return gated_x


class GetOrthogonal(nn.Module):
    """
    Get orthogonal components of every vector of a sequence
    compared to the corresponding vectors of an another sequence.

    (xi+) = ((xi . yi)/(yi . yi)) * yi
    (xi-) = xi - (xi+)
    """

    def __init__(self, x_size, y_size, identity=False):
        super(GetOrthogonal, self).__init__()

        # If identity is true, we just use a dot product without transformation.
        if not identity:
            self.linear = nn.Linear(y_size, x_size)
        else:
            self.linear = None

    def forward(self, x, y):
        """
        Args:
            x: batch * len * hdim1
            y: batch * len * hdim2

        Output:
            alpha_x = batch * hdim1
        """
        y_flat = y.view(-1, y.size(-1))
        Wy_flat = self.linear(y_flat) if self.linear is not None else y_flat
        Wy = Wy_flat.view(y.size(0), y.size(1), x.size(-1))
        xWy = x.bmm(Wy.transpose(2, 1))  # batch * len * len
        xi_dot_yis = xWy.view(xWy.size(0), -1)[:, ::xWy.size(1) + 1]  # getting the diagonals: batch * len
        WyWy = Wy.bmm(Wy.transpose(2, 1))  # batch * len * len
        yi_dot_yis = WyWy.view(WyWy.size(0), -1)[:, ::WyWy.size(1) + 1]
        xiyi_div_yiyi = torch.div(xi_dot_yis, yi_dot_yis)
        xi_plus = torch.mul(xiyi_div_yiyi.unsqueeze(2), Wy)  # parallel components : batch * len * hdim1
        xi_minus = torch.add(x, torch.mul(xi_plus, -1))     # orthogonal components : batch * len * hdim1

        return xi_plus, xi_minus


class GatedMultifactorSelfAttnEnc(nn.Module):
    """Gated multi-factor self attentive encoding over a sequence:
    """
    def __init__(self, input_size, num_factor, attn_pooling='max'):
        super(GatedMultifactorSelfAttnEnc, self).__init__()
        self.num_factor = num_factor
        if self.num_factor > 0:
            self.linear = nn.Linear(input_size, self.num_factor * input_size)
        else:
            self.linear = None
        self.linear_gate = nn.Linear(2 * input_size, 2 * input_size)
        self.attn_pooling = attn_pooling

    def forward(self, x, x_mask):
        """
        Args:
            x: batch * len * hdim
        Output:
            gated_selfmatched_x: batch * len * 2hdim
        """
        if self.linear is not None:
            self_attn_multi = []
            y_multi = self.linear(x.view(-1, x.size(2))).view(x.size(0), x.size(1),
                                                        x.size(2) * self.num_factor)
            y_multi = F.relu(y_multi)
            y_multi = y_multi.view(x.size(0), x.size(1), x.size(2), self.num_factor)
            for fac in range(self.num_factor):
                # foo = str(fac)
                # exec('y' + foo + " = y_multi.narrow(2, fac, 1)")
                # exec('attn_fac = y' + foo + '.bmm(y' + foo + '.transpose(2,1)')
                y = y_multi.narrow(3, fac, 1).squeeze()
                attn_fac = y.bmm(y.transpose(2,1))
                attn_fac = attn_fac.unsqueeze(-1)
                self_attn_multi.append(attn_fac)
            self_attn_multi = torch.cat(self_attn_multi, -1) # batch * len * len *  num_factor

            #
            # y = self.linear(x.view(-1, x.size(2))).view(x.size(0), x.size(1),
            #                                             self.num_factor, x.size(2))
            #
            # # calculate the self attn mat
            # self_attn_multi = y.view(y.size(0), -1,
            #                         y.size(3)).bmm(x.transpose(2,1)).view(y.size(0),
            #                         y.size(1), y.size(2), x.size(1))  # batch * len * num_factor * len
            if self.attn_pooling == 'max':
                self_attn, _ = torch.max(self_attn_multi, 3)  # batch * len * len
            elif self.attn_pooling == 'min':
                self_attn, _ = torch.min(self_attn_multi, 3)
            else:
                self_attn = torch.mean(self_attn_multi, 3)
        else:
            self_attn = x.bmm(x.transpose(2,1))  # batch * len * len

        x_mask_repeat = x_mask.repeat(1, x_mask.size(1))
        x_mask_repeat = x_mask_repeat.view(x_mask.size(0), x_mask.size(1), -1)
        self_attn.data.masked_fill_(x_mask_repeat.data, -float('inf'))

        #  make the diagonal elements to -inf so that it does not consider the same word
        self_attn_mask = torch.diag(torch.ones(self_attn.size(1)) * -float('inf'))
        # self_attn_mask = torch.diag(torch.cuda.FloatTensor([1]*self_attn.size(1)) * -float('inf'))
        if self_attn.data.is_cuda:
            self_attn_mask = self_attn_mask.cuda()
        self_attn_mask = self_attn_mask.repeat(self_attn.size(0), 1).view(self_attn.size())
        self_attn.data.add_(self_attn_mask)

        # Normalize with softmax
        alpha_flat = F.softmax(self_attn.view(-1, x.size(1)))
        alpha = alpha_flat.view(-1, x.size(1), x.size(1))  # batch * len * len

        # multifactor attentive enc
        multi_attn_enc = alpha.bmm(x)  # batch * len * hdim

        # merge with original x
        gate_input = [x]
        gate_input.append(multi_attn_enc)
        joint_ctx_input = torch.cat(gate_input, 2)

        # gating
        gate_joint_ctx_self_match = self.linear_gate(joint_ctx_input.view(-1, joint_ctx_input.size(2))).view(joint_ctx_input.size())
        gate_joint_ctx_self_match = F.sigmoid(gate_joint_ctx_self_match)

        gated_multi_attentive_enc = torch.mul(gate_joint_ctx_self_match, joint_ctx_input)

        return gated_multi_attentive_enc


class ConvCharEmb(nn.Module):
    """
    CNN-Maxpooling based character embedding
    """

    def __init__(self, input_size, num_filter, filter_length, pooling='maxpool'):
        super(ConvCharEmb, self).__init__()
        self.num_filter = num_filter
        self.filter_length = filter_length
        self.pooling = pooling
        self.char_cnn = nn.Conv1d(input_size, num_filter,
                                  filter_length)


    def forward(self, x):
        """
        Args:
            x: (batch * len) * maxwordlen * charembdim
        Output:
            char_cnn_x: (batch * len) * num_filter
        """
        y = self.char_cnn(x.transpose(2,1))
        if self.pooling == 'maxpool':
            maxpl_y, _ = torch.max(y, 2)
        else:
            raise RuntimeError('Not implemented this pooling!')
        return maxpl_y


class ConvSeqEncPool(nn.Module):
    """
    CNN-Pooling based sequence encoding
    """

    def __init__(self, input_size, num_filter, filter_length, pooling='maxpool'):
        super(ConvSeqEncPool, self).__init__()
        self.num_filter = num_filter
        self.filter_length = filter_length
        self.pooling = pooling
        self.char_cnn = nn.Conv1d(input_size, num_filter,
                                  filter_length)


    def forward(self, x):
        """
        Args:
            x: batch * len * hdim
        Output:
            char_cnn_x: batch * num_filter
        """
        y = self.char_cnn(x.transpose(2,1))
        y = F.relu(y)
        if self.pooling == 'maxpool':
            pl_y, _ = torch.max(y, 2)
        if self.pooling == 'meanpool':
            pl_y = torch.mean(y, 2)
        else:
            raise RuntimeError('Not implemented this pooling!')
        return pl_y


class ConvSeqEnc(nn.Module):
    """
    CNN-Pooling based sequence encoding
    """

    def __init__(self, input_size, num_filter, filter_length):
        super(ConvSeqEnc, self).__init__()
        self.num_filter = num_filter
        self.filter_length = filter_length
        self.seq_cnn = nn.Conv1d(input_size, num_filter,
                                  filter_length)


    def forward(self, x):
        """
        Args:
            x: batch * len * hdim
        Output:
            y: batch * len2 * num_filter
        """
        y = self.seq_cnn(x.transpose(2,1))
        y = y.transpose(2,1)
        return y


class LinearLayer(nn.Module):
    """
    Linear layer
    """
    def __init__(self, input_size, output_size, activation='tanh'):
        super(LinearLayer, self).__init__()
        self.linear = nn.Linear(input_size, output_size)
        self.activation = activation

    def forward(self, x):
        """
        Args:
            x: batch * hdim1
        Output:
            y: batch * hdim2
        """
        y = self.linear(x)
        if self.activation == 'tanh':
            y = F.tanh(y)
        elif self.activation == 'relu':
            y = F.relu(y)

        return y


class LinearLayerSeq(nn.Module):
    """
    Linear layer
    """
    def __init__(self, input_size, output_size, activation='tanh'):
        super(LinearLayerSeq, self).__init__()
        self.linear = nn.Linear(input_size, output_size)
        self.activation = activation

    def forward(self, x):
        """
        Args:
            x: batch * len * hdim1
        Output:
            y: batch * len * hdim2
        """
        x_flat = x.view(-1, x.size(2))
        y = self.linear(x_flat)
        y = y.view(x.size(0), x.size(1), -1)
        if self.activation == 'tanh':
            y = F.tanh(y)
        elif self.activation == 'relu':
            y = F.relu(y)
        elif self.activation == 'sigmoid':
            y = F.sigmoid(y)

        return y


class LinearSeqAttnAggrt(nn.Module):
    """Self attention over a sequence:

    * o_i = softmax(Wx_i) for x_i in X.
    """

    def __init__(self, input_size):
        super(LinearSeqAttnAggrt, self).__init__()
        self.linear = nn.Linear(input_size, 1)

    def forward(self, x, x_mask):
        """
        Args:
            x: batch * len * hdim
            x_mask: batch * len (1 for padding, 0 for true)
        Output:
            alpha: batch * len
        """
        x_flat = x.view(-1, x.size(-1))
        scores = self.linear(x_flat).view(x.size(0), x.size(1))
        scores.data.masked_fill_(x_mask.data, -float('inf'))
        alpha = F.softmax(scores)
        enc = alpha.unsqueeze(1).bmm(x).squeeze(1)
        return enc


class SeqAttnMatchSoftmin(nn.Module):
    """Given sequences X and Y, match sequence Y to each element in X.

    * o_i = sum(alpha_j * y_j) for i in X
    * alpha_j = softmax(y_j * x_i)
    """

    def __init__(self, input_size, identity=True):
        super(SeqAttnMatchSoftmin, self).__init__()
        if not identity:
            self.linear = nn.Linear(input_size, input_size)
        else:
            self.linear = None

    def forward(self, x, y, y_mask):
        """
        Args:
            x: batch * len1 * hdim
            y: batch * len2 * hdim
            y_mask: batch * len2 (1 for padding, 0 for true)
        Output:
            matched_seq: batch * len1 * hdim
        """
        # Project vectors
        if self.linear:
            x_proj = self.linear(x.view(-1, x.size(2))).view(x.size())
            x_proj = F.relu(x_proj)
            y_proj = self.linear(y.view(-1, y.size(2))).view(y.size())
            y_proj = F.relu(y_proj)
        else:
            x_proj = x
            y_proj = y

        # Compute scores
        scores = x_proj.bmm(y_proj.transpose(2, 1))

        # Mask padding
        y_mask = y_mask.unsqueeze(1).expand(scores.size())
        scores.data.masked_fill_(y_mask.data, float('inf'))

        # Normalize with softmin
        alpha_flat = F.softmin(scores.view(-1, y.size(1)))
        alpha = alpha_flat.view(-1, x.size(1), y.size(1))

        # Take weighted average
        matched_seq = alpha.bmm(y)
        return matched_seq


class BilinearSeqAttnBoth(nn.Module):
    """A bilinear attention layer over a sequence X w.r.t y:

    * o_i = softmax(x_i'Wy) for x_i in X.

    Optionally don't normalize output weights.
    """

    def __init__(self, x_size, y_size, identity=False, normalize=True):
        super(BilinearSeqAttnBoth, self).__init__()
        self.normalize = normalize

        # If identity is true, we just use a dot product without transformation.
        if not identity:
            self.linear = nn.Linear(y_size, x_size)
        else:
            self.linear = None

    def forward(self, x, y, x_mask):
        """
        Args:
            x: batch * len * hdim1
            y: batch * hdim2
            x_mask: batch * len (1 for padding, 0 for true)
        Output:
            alpha = batch * len
        """
        Wy = self.linear(y) if self.linear is not None else y
        xWy = x.bmm(Wy.unsqueeze(2)).squeeze(2)
        unnorm_scores = xWy.clone()
        xWy.data.masked_fill_(x_mask.data, -float('inf'))
        if self.normalize:
            if self.training:
                # In training we output log-softmax for NLL
                alpha = F.log_softmax(xWy)
            else:
                # ...Otherwise 0-1 probabilities
                alpha = F.softmax(xWy)
        else:
            alpha = xWy.exp()
        return unnorm_scores, alpha


class HighwayLayer(nn.Module):
    """
    Highway Layer
    """

    def __init__(self, input_size, output_size):
        super(HighwayLayer, self).__init__()
        assert input_size == output_size
        self.trans_lin = nn.Linear(input_size, output_size)
        self.gate_lin = nn.Linear(input_size, output_size)

    def forward(self, x):
        """
        Args:
            x: batch * len * hdim
        Output:
            y: batch * len * hdim
        y = gate * trans + (1-gate) * x
        """
        trans = self.trans_lin(x)
        trans = F.relu(trans)
        gate = self.gate_lin(x)
        gate = F.sigmoid(gate)
        ones_tensor = Variable(torch.ones(gate.size()).cuda(async=True))
        one_minus_gate = torch.add(ones_tensor, torch.mul(gate, -1))
        y = torch.add(torch.mul(gate, trans), torch.mul(one_minus_gate, x))
        return y


class SimilarityLayerBiDAF(nn.Module):
    """
    similarity matrix S of BiDAF
    """

    def __init__(self, input_size):
        super(SimilarityLayerBiDAF, self).__init__()
        self.attn_linear = nn.Linear(3 * input_size, 1)

    def forward(self, x1, x2, x2_mask):
        """
        alpha(h,u) = w[h, u, h.u]
        :param x1: batch * len1 * hdim
        :param x2: batch * len2 * hdim
        :x2_mask: batch * len2
        :return: batch * len1 * len2
        """
        repeat_x1 = x1.unsqueeze(2).expand(x1.size(0), x1.size(1), x2.size(1), x1.size(2))
        repeat_x2 = x2.unsqueeze(1).expand(x2.size(0), x1.size(1), x2.size(1), x2.size(2))
        rx1rx2 = torch.mul(repeat_x1, repeat_x2)
        combined = torch.cat([repeat_x1, repeat_x2, rx1rx2], 3)
        combined_flat = combined.view(-1, combined.size(3))
        flat_attn = self.attn_linear(combined_flat)
        flat_attn = flat_attn.view(x1.size(0), x1.size(1), x2.size(1), 1)
        scores = flat_attn.squeeze(3)  # batch * len1 * len2
        # Mask padding
        x2_mask = x2_mask.unsqueeze(1).expand(scores.size())
        scores.data.masked_fill_(x2_mask.data, -float('inf'))

        # Normalize with softmax
        alpha_flat = F.softmax(scores.view(-1, x2.size(1)))
        alpha = alpha_flat.view(-1, x1.size(1), x2.size(1))
        return scores, alpha


class LinearSeqAttnFinal(nn.Module):
    """Self attention over a sequence:

    * o_i = softmax(Wx_i) for x_i in X.
    """

    def __init__(self, input_size, normalize=True):
        super(LinearSeqAttnFinal, self).__init__()
        self.normalize = normalize
        self.linear = nn.Linear(input_size, 1)

    def forward(self, x, x_mask):
        """
        Args:
            x: batch * len * hdim
            x_mask: batch * len (1 for padding, 0 for true)
        Output:
            alpha: batch * len
        """
        x_flat = x.view(-1, x.size(-1))
        scores = self.linear(x_flat).view(x.size(0), x.size(1))
        scores_unnorm = scores.clone()
        scores.data.masked_fill_(x_mask.data, -float('inf'))

        if self.normalize:
            if self.training:
                # In training we output log-softmax for NLL
                alpha = F.log_softmax(scores)
            else:
                # ...Otherwise 0-1 probabilities
                alpha = F.softmax(scores)
        else:
            alpha = scores.exp()
        return scores_unnorm, alpha


class LinearSeqAttnFinalNil(nn.Module):
    """Self attention over a sequence:

    * o_i = softmax(Wx_i) for x_i in X.
    """
    def __init__(self, input_size, nil_input_size, normalize=True):
        super(LinearSeqAttnFinalNil, self).__init__()
        self.normalize = normalize
        self.linear = nn.Linear(input_size, 1)
        self.nil_linear = nn.Linear(nil_input_size, 1)

    def forward(self, x, nil_vec, x_mask):
        """
        Args:
            x: batch * len * hdim
            nil_vec: batch * 1 * hdim
            x_mask: batch * len (1 for padding, 0 for true)
        Output:
            alpha: batch * len
        """
        x_flat = x.view(-1, x.size(-1))
        x_scores = self.linear(x_flat).view(x.size(0), x.size(1))
        nil_vec_flat = nil_vec.view(-1, nil_vec.size(-1))
        nil_score = self.nil_linear(nil_vec_flat).view(nil_vec.size(0), 1)
        scores = torch.cat([nil_score, x_scores], 1)
        scores_unnorm = scores.clone()
        scores.data.masked_fill_(x_mask.data, -float('inf'))

        if self.normalize:
            if self.training:
                # In training we output log-softmax for NLL
                alpha = F.log_softmax(scores)
            else:
                # ...Otherwise 0-1 probabilities
                alpha = F.softmax(scores)
        else:
            alpha = scores.exp()
        return scores_unnorm, alpha
# ------------------------------------------------------------------------------
# Functional
# ------------------------------------------------------------------------------


def uniform_weights(x, x_mask):
    """Return uniform weights over non-masked x (a sequence of vectors).

    Args:
        x: batch * len * hdim
        x_mask: batch * len (1 for padding, 0 for true)
    Output:
        x_avg: batch * hdim
    """
    alpha = Variable(torch.ones(x.size(0), x.size(1)))
    if x.data.is_cuda:
        alpha = alpha.cuda()
    alpha = alpha * x_mask.eq(0).float()
    alpha = alpha / alpha.sum(1).expand(alpha.size())
    return alpha


def weighted_avg(x, weights):
    """Return a weighted average of x (a sequence of vectors).

    Args:
        x: batch * len * hdim
        weights: batch * len, sum(dim = 1) = 1
    Output:
        x_avg: batch * hdim
    """
    return weights.unsqueeze(1).bmm(x).squeeze(1)


def sequential_weighted_avg(x, weights):
    """Return a sequence by weighted averaging of x (a sequence of vectors).

    Args:
        x: batch * len2 * hdim
        weights: batch * len1 * len2, sum(dim = 2) = 1
    Output:
        x_avg: batch * len1 * hdim
    """
    # return weights.unsqueeze(1).bmm(x).squeeze(1)
    return weights.bmm(x)


def mean_over_time(x):
    s = torch.mean(x, 1)
    return s


def mean_over_time_masked(x, x_mask):
    """Return uniform weights over non-masked x (a sequence of vectors).

    Args:
        x: batch * len * hdim
        x_mask: batch * len (1 for padding, 0 for true)
    Output:
        x_avg: batch * hdim
    """
    alpha = Variable(torch.ones(x.size(0), x.size(1)))
    if x.data.is_cuda:
        alpha = alpha.cuda()
    alpha = alpha * x_mask.eq(0).float()
    # alpha = alpha / alpha.sum(1).expand(alpha.size())
    alpha = alpha / alpha.sum(1).unsqueeze(1).repeat(1, alpha.size(1))

    mot_x = alpha.unsqueeze(1).bmm(x).squeeze(1)
    return mot_x


def max_attentional_aggregation(x, x_mask, weights):
    """Return a max attentional aggregation based on
    max attentive pooling.

    Args:
        x: batch * len2 * hdim
        x_mask: batch * len (1 for padding, 0 for true)
        weights: batch * len1 * len2, sum(dim = 2) = 1
    Output:
        x_avg: batch * hdim
    """
    # return weights.unsqueeze(1).bmm(x).squeeze(1)
    # max_attn = weights.max(1)  # batch * len2
    max_attn, _ = torch.max(weights, 1)  # batch * len2
    max_attn.data.masked_fill_(x_mask.data, -float('inf'))
    alpha = F.softmax(max_attn)
    return alpha.unsqueeze(1).bmm(x).squeeze(1)


def get_qwh2(x, x_f):
    """Return the t_wh and next to t_wh
    word vectors of a question.

	Args:
	    x: batch * len2 * hdim
	    x_f: batch * len2 * 2
	Output:
	    x_wh2: batch * (2 * hdim)
    """
    q_wh2 = x_f.transpose(2,1).bmm(x)  # batch * 2 * hdim
    return q_wh2.view(q_wh2.size(0), -1)


def apply_activation(x, activation='tanh'):
    if activation == 'tanh':
        return F.tanh(x)
    elif activation == 'relu':
        return F.relu(x)
    elif activation == 'sigmoid':
        return F.sigmoid(x)
    elif activation == 'softmax':
        return F.softmax(x)
    else:
        raise RuntimeError('Not implemented this activation!')
