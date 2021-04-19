from typing import Union

import torch
from torch import nn
import torch.nn.init as init
import math
from scipy.special import comb as nchoosek

from config import INF


class WSDModel(nn.Module):

    def __init__(self, V, Y, D: int = 300, dropout_prob: float = 0.2, use_padding: bool = False,
                 use_positional_encodings: bool = False, pos_exponent: int = 1, pos_cutoff_position: int = 5,
                 pos_is_causal: bool = False, pos_normalize_magnitude: bool = False):
        super(WSDModel, self).__init__()
        self.use_padding = use_padding

        self.D = D
        self.pad_id = 0
        self.E_v = nn.Embedding(V, D, padding_idx=self.pad_id)
        self.E_y = nn.Embedding(Y, D, padding_idx=self.pad_id)
        init.kaiming_uniform_(self.E_v.weight[1:], a=math.sqrt(5))
        init.kaiming_uniform_(self.E_y.weight[1:], a=math.sqrt(5))

        self.W_A = nn.Parameter(torch.Tensor(D, D))
        self.W_O = nn.Parameter(torch.Tensor(D, D))
        init.kaiming_uniform_(self.W_A, a=math.sqrt(5))
        init.kaiming_uniform_(self.W_O, a=math.sqrt(5))

        self.dropout_layer = nn.Dropout(p=dropout_prob)
        self.softmax = torch.nn.Softmax(dim=-1)
        self.layer_norm = nn.LayerNorm([self.D])

        self.use_positional_encodings = use_positional_encodings
        self.pos_exponent = pos_exponent
        self.pos_cutoff_position = pos_cutoff_position
        self.pos_is_causal = pos_is_causal
        self.pos_normalize_magnitude = pos_normalize_magnitude

    def attention(self, X, Q, mask):
        """
        Computes the contextualized representation of query Q, given context X, using the attention model.

        :param X:
            Context matrix of shape [B, N, D]
        :param Q:
            Query matrix of shape [B, k, D], where k equals 1 (in single word attention) or N (self attention)
        :param mask:
            Boolean mask of size [B, N] indicating padding indices in the context X.

        :return:
            Contextualized query and attention matrix / vector
        """
        # TODO Part 1: Your code here.
        # Have a look at the difference between torch.matmul() and torch.bmm().      
        logits = torch.bmm(Q @ self.W_A, X.transpose(1, 2))

        if self.use_positional_encodings:
            pos_rep = self.generate_positional_representations(seq_len=X.shape[1],
                                                               exponent=self.pos_exponent,
                                                               cutoff_position=self.pos_cutoff_position,
                                                               is_causal=self.pos_is_causal,
                                                               device=logits.device)
            if self.pos_normalize_magnitude:
                not_inf = (~torch.isinf(pos_rep)) & (pos_rep != -INF)
                pos_magnitude = torch.max(torch.abs(pos_rep[not_inf]))
                logits_magnitude = torch.max(torch.abs(logits))

                pos_rep_norm = pos_rep / pos_magnitude * logits_magnitude
                pos_rep = torch.where(not_inf, pos_rep_norm, torch.Tensor([-INF]).to(pos_rep.device))

                # not_inf = (~torch.isinf(pos_rep)) & (pos_rep != -INF)
                # pos_magnitude_after_normalization = torch.max(torch.abs(pos_rep[not_inf]))
                # print(pos_magnitude.item(), logits_magnitude.item(), pos_magnitude_after_normalization.item())

            logits = logits + pos_rep

        if self.use_padding:
            # TODO part 2: Your code here.
            logits = logits.masked_fill(mask.unsqueeze(1) == self.pad_id, -INF)

        # TODO Part 1: continue.
        A = self.softmax(logits)
        Q_c = torch.bmm(A, X @ self.W_O)

        if torch.rand(1).item() < 0.01:
            print("\nA\n", A[0, :5, :5])
            if self.use_positional_encodings:
                print("\npos_rep\n", pos_rep[:5, :5])

        return Q_c, A.squeeze()

    def forward(self, M_s, v_q=None):
        """
        :param M_s:
            [B, N] dimensional matrix containing token integer ids
        :param v_q:
            [B] dimensional vector containing query word indices within the sentences represented by M_s.
            This argument is only passed in single word attention mode.

        :return: logits and attention tensors.
        """

        X = self.dropout_layer(self.E_v(M_s))  # [B, N, D]

        Q = None
        if v_q is not None:
            # TODO Part 1: Your Code Here.
            v_q = v_q.reshape(-1, 1, 1)
            v_q = v_q.expand(v_q.shape[0], 1, self.D)
            Q = torch.gather(X, 1, v_q)
        else:
            # TODO Part 3: Your Code Here.
            Q = X

        mask = M_s.ne(self.pad_id)
        Q_c, A = self.attention(X, Q, mask)
        H = self.layer_norm(Q_c + Q)

        E_y = self.dropout_layer(self.E_y.weight)
        y_logits = (H @ E_y.T).squeeze()
        return y_logits, A

    @staticmethod
    def generate_positional_representations(seq_len: int,
                                            exponent: int = 1,
                                            cutoff_position: int = 5,
                                            is_causal: bool = False,
                                            device: Union[str, torch.device] = "cpu") -> torch.Tensor:
        relative_positions = torch.arange(seq_len).unsqueeze(0) - torch.arange(seq_len).unsqueeze(1)
        relative_positions = relative_positions.to(device)

        abs_relative_positions = torch.abs(relative_positions).float()
        abs_relative_positions = torch.where(abs_relative_positions < cutoff_position, abs_relative_positions,
                                             torch.Tensor([cutoff_position]).to(device))
        pos_rep = - abs_relative_positions ** exponent

        if is_causal:
            is_future = relative_positions > 0
            # minus_inf = torch.Tensor([-float("inf")]).to(device)
            minus_inf = torch.Tensor([-INF]).to(device)
            pos_rep = torch.where(is_future, minus_inf, pos_rep)

        return pos_rep

    def generate_position_factors_2(self, seq_len: int, window_size: int = 3,
                                    kernel_type: str = "binomial",
                                    use_zero_factors_for_distant_words: bool = True) -> torch.Tensor:
        """
        :param kernel_type:
            "binomial": using binomial coefficients to approximate a gaussian distribution.
            "mean": a simple mean kernel e.g. [0.333, 0.333, 0.333]
        """
        kernel_size = 2 * window_size + 1
        if kernel_type == "binomial":
            weights_kernel = [nchoosek(kernel_size - 1, k) for k in range(0, kernel_size)]
        elif kernel_type == "mean":
            weights_kernel = [1] * kernel_size
        weights_kernel = torch.Tensor(weights_kernel)
        weights_kernel = weights_kernel / weights_kernel.sum()

        if use_zero_factors_for_distant_words:
            single_pos_rep = torch.zeros(seq_len)
        else:
            single_pos_rep = torch.ones(seq_len) * weights_kernel.min()

        single_pos_rep = torch.cat([weights_kernel, single_pos_rep])

        all_pos_rep = []
        for i in range(window_size + 1, seq_len + window_size + 1):
            curr_pos_rep = torch.roll(single_pos_rep, i)
            curr_pos_rep = curr_pos_rep[kernel_size:]
            curr_pos_rep = curr_pos_rep.unsqueeze(1)
            all_pos_rep.append(curr_pos_rep)

        pos_rep_matrix = torch.cat(all_pos_rep, dim=1)
        return pos_rep_matrix
