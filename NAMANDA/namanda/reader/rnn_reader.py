#!/usr/bin/env python3
"""Implementation of NAMANDA"""

import torch
import torch.nn as nn
from . import layers


# ------------------------------------------------------------------------------
# Network
# ------------------------------------------------------------------------------


class RnnDocReader(nn.Module):
    '''Linear Aggregator
    Final Model EMNLP 2018
    '''
    RNN_TYPES = {'lstm': nn.LSTM, 'gru': nn.GRU, 'rnn': nn.RNN}

    def __init__(self, args, normalize=True):
        super(RnnDocReader, self).__init__()
        # Store config
        self.args = args

        # Word embeddings (+1 for padding)
        self.embedding = nn.Embedding(args.vocab_size,
                                      args.embedding_dim,
                                      padding_idx=0)

        if self.args.char_embedding:
            self.char_embedding = nn.Embedding(args.char_vocab_size,
                                               args.char_embeddim,
                                               padding_idx=0)

            self.char_cnn = layers.ConvCharEmb(args.char_embeddim, args.char_embeddim, 5)

            doc_input_size = args.embedding_dim + args.num_features + args.char_embeddim
            q_input_size = args.embedding_dim + args.char_embeddim
        else:
            # Input size to RNN: word emb + manual features
            doc_input_size = args.embedding_dim + args.num_features
            # doc_input_size = args.embedding_dim
            q_input_size = args.embedding_dim

        # assert doc_input_size == q_input_size
        # sequence level passage encoder
        self.seqlevelenc_rnn = layers.StackedBRNN(
            input_size=doc_input_size,
            hidden_size=args.hidden_size,
            num_layers=args.doc_layers, ## always should be 1
            dropout_rate=args.dropout_rnn,
            dropout_output=args.dropout_rnn_output,
            concat_layers=args.concat_rnn_layers,
            rnn_type=self.RNN_TYPES[args.rnn_type],
            padding=args.rnn_padding,
        )

        # Output sizes of rnn encoders
        doc_hidden_size = 2 * args.hidden_size
        question_hidden_size = 2 * args.hidden_size
        if args.concat_rnn_layers:
            doc_hidden_size *= args.doc_layers
            question_hidden_size *= args.question_layers

        assert doc_hidden_size == question_hidden_size

        # cartesian similarity based attention layer
        self.cartesian_attn = layers.SeqAttnMat(doc_hidden_size, identity=True)

        # encoding for question dep passage rep
        self.gate_qdep_penc = layers.GatedEncoding(doc_hidden_size + question_hidden_size)

        # question dependent passage encoding rnn
        self.qdep_penc_rnn = layers.StackedBRNN(
            input_size=doc_hidden_size + question_hidden_size,
            hidden_size=args.hidden_size,
            num_layers=args.doc_layers,  ## always should be 1
            dropout_rate=args.dropout_rnn,
            dropout_output=args.dropout_rnn_output,
            concat_layers=args.concat_rnn_layers,
            rnn_type=self.RNN_TYPES[args.rnn_type],
            padding=args.rnn_padding,
        )

        # multi-factor attentive encoding
        self.multifactor_attn = layers.GatedMultifactorSelfAttnEnc(
            input_size=doc_hidden_size, num_factor=args.num_factor)

        # ortho decomp
        self.orthodecomp = layers.GetOrthogonal(2 * doc_hidden_size,
                                                2 * doc_hidden_size,
                                                identity=True)

        # nil linear
        self.nil_linear = layers.LinearLayerSeq(2*doc_hidden_size,
                                                doc_hidden_size)

        # rnn for answer starting pointer
        self.ans_start_rnn = layers.StackedBRNN(
            input_size=2 * doc_hidden_size,
            hidden_size=args.hidden_size,
            num_layers=1,  ## always should be 1
            dropout_rate=args.dropout_rnn,
            dropout_output=args.dropout_rnn_output,
            concat_layers=args.concat_rnn_layers,
            rnn_type=self.RNN_TYPES[args.rnn_type],
            padding=args.rnn_padding,
        )

        # rnn for answer ending pointer
        self.ans_end_rnn = layers.StackedBRNN(
            input_size=doc_hidden_size,
            hidden_size=args.hidden_size,
            num_layers=1,  ## always should be 1
            dropout_rate=args.dropout_rnn,
            dropout_output=args.dropout_rnn_output,
            concat_layers=args.concat_rnn_layers,
            rnn_type=self.RNN_TYPES[args.rnn_type],
            padding=args.rnn_padding,
        )

        # linear layer for q_tilde
        self.linear_qtilde = layers.LinearLayer(
            3 * question_hidden_size,
            question_hidden_size
        )

        # layer for answer start/end
        self.start_end_attn = layers.BilinearSeqAttnwithNIL(
            doc_hidden_size,
            question_hidden_size,
            normalize=normalize,
        )

    def forward(self, x1, x1_f, x1_char, x1_mask, x2, x2_char, x2_mask):
        """Inputs:
        x1 = document word indices             [batch * len_d]
        x1_f = document word features indices  [batch * len_d * nfeat]
        x1_mask = document padding mask        [batch * len_d]
        x2 = question word indices             [batch * len_q]
        x2_mask = question padding mask        [batch * len_q]
        """

        # Embed both document and question
        x1_emb = self.embedding(x1)
        x2_emb = self.embedding(x2)

        # Dropout on embeddings
        if self.args.dropout_emb > 0:
            x1_emb = nn.functional.dropout(x1_emb, p=self.args.dropout_emb,
                                           training=self.training)
            x2_emb = nn.functional.dropout(x2_emb, p=self.args.dropout_emb,
                                           training=self.training)

        # Form document encoding inputs
        drnn_input = [x1_emb]
        qrnn_input = [x2_emb]

        # Add char embedding
        if self.args.char_embedding:
            x1_char_flat = x1_char.view(-1, self.args.maxwordlen)
            x1_flat_char_emb = self.char_embedding(x1_char_flat)  # (batch*len) * maxwordlen * charembdim
            x2_char_flat = x2_char.view(-1, self.args.maxwordlen)
            x2_flat_char_emb = self.char_embedding(x2_char_flat)
            x1_char_cnn = self.char_cnn(x1_flat_char_emb)
            x2_char_cnn = self.char_cnn(x2_flat_char_emb)
            x1_char_cnn = x1_char_cnn.view(x1.size(0), x1.size(1), -1)
            x2_char_cnn = x2_char_cnn.view(x2.size(0), x2.size(1), -1)
            drnn_input.append(x1_char_cnn)
            qrnn_input.append(x2_char_cnn)

        # Add manual features
        # if self.args.num_features > 0:
        #     drnn_input.append(x1_f)

        drnn_input = torch.cat(drnn_input, 2)
        drnn_input = drnn_input.narrow(1, 1, drnn_input.size(1) - 1)
        nil_mask = x1_mask.narrow(1, 0, 1)
        orig_x1_mask = x1_mask.narrow(1, 1, x1_mask.size(1) - 1)

        # Encode document with RNN
        doc_hiddens = self.seqlevelenc_rnn(drnn_input, orig_x1_mask)

        # Encode question with RNN
        question_hiddens = self.seqlevelenc_rnn(torch.cat(qrnn_input, 2), x2_mask)

        q_dep_passage_enc_rnn_input = [doc_hiddens]

        # attn mat calculation
        unnorm_cartesian_attn_mat, normalized_cartesian_attn_mat = self.cartesian_attn(doc_hiddens,
                                                                                       question_hiddens,
                                                                                       x2_mask)

        # question dependeent passage encoding
        q_aware_passage_rep = layers.sequential_weighted_avg(question_hiddens,
                                                             normalized_cartesian_attn_mat)
        q_dep_passage_enc_rnn_input.append(q_aware_passage_rep)

        # gated question dependent passage encoding
        gated_qaware_passage_rep = self.gate_qdep_penc(torch.cat(q_dep_passage_enc_rnn_input, 2))
        qdep_penc_hiddens = self.qdep_penc_rnn(gated_qaware_passage_rep, orig_x1_mask)

        # multi factor attentive encoding
        multi_attentive_enc = self.multifactor_attn(qdep_penc_hiddens, orig_x1_mask)

        # orthogonal decomposition of MFA enc
        mfa_enc_plus, mfa_enc_minus = self.orthodecomp(multi_attentive_enc,
                                                       gated_qaware_passage_rep)
        # mfa_nil_conv_in = torch.cat([mfa_enc_plus, mfa_enc_minus], 2)
        mfa_nil_linear_plus = self.nil_linear(mfa_enc_plus)
        mfa_nil_linear_minus = self.nil_linear(mfa_enc_minus)
        mfa_nil_lin_comb = torch.add(mfa_nil_linear_plus, mfa_nil_linear_minus)
        mfa_nil_conv, _ = torch.max(mfa_nil_lin_comb, 1)
        mfa_nil_conv = mfa_nil_conv.unsqueeze(1)

        # rnn encoding for answer starting pointer
        ans_start_enc_hiddens = self.ans_start_rnn(multi_attentive_enc, orig_x1_mask)

        # rnn encoding for answer ending pointer
        ans_end_enc_hiddens = self.ans_end_rnn(ans_start_enc_hiddens, orig_x1_mask)

        ans_start_enc_hiddens = torch.cat([mfa_nil_conv, ans_start_enc_hiddens], 1)
        ans_end_enc_hiddens = torch.cat([mfa_nil_conv, ans_end_enc_hiddens], 1)

        # max attentional question aggregation
        q_ma = layers.max_attentional_aggregation(question_hiddens, x2_mask,
                                                  unnorm_cartesian_attn_mat)
        # question focus
        q_f = layers.get_qwh2(question_hiddens, x1_f)
        q_tilde = self.linear_qtilde(torch.cat([q_ma, q_f], 1))

        # Predict start and end positions
        start_scores = self.start_end_attn(ans_start_enc_hiddens, q_tilde, x1_mask)
        end_scores = self.start_end_attn(ans_end_enc_hiddens, q_tilde, x1_mask)
        return start_scores, end_scores
