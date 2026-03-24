"""BiLSTM+CRF model for Korean morphological analysis.

Jamo-level sequence labelling with BIO tags.
"""

import torch
import torch.nn as nn
from typing import List, Optional, Tuple


class BiLstmCrf(nn.Module):
    """Bidirectional LSTM with a CRF output layer.

    Args:
        vocab_size: Number of input token IDs (57 for Jamo vocab).
        embed_dim: Embedding dimension.
        hidden_size: LSTM hidden size per direction.
        num_tags: Number of BIO tag labels.
        num_layers: Number of stacked BiLSTM layers.
        dropout: Dropout rate between LSTM layers.
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        hidden_size: int,
        num_tags: int,
        num_layers: int = 2,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.hidden_size = hidden_size
        self.num_tags = num_tags
        self.num_layers = num_layers

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        # Output projection from BiLSTM hidden (hidden_size*2) -> num_tags
        self.output_proj = nn.Linear(hidden_size * 2, num_tags)

        # CRF transition matrix: transitions[i, j] = score of tag j -> tag i
        self.transitions = nn.Parameter(torch.randn(num_tags, num_tags))
        # Constrain start / end: no special start/end tags needed for BIO,
        # but initialise transitions to reasonable values.
        nn.init.uniform_(self.transitions, -0.1, 0.1)

    # -----------------------------------------------------------------
    # Emissions
    # -----------------------------------------------------------------

    def forward_emissions(self, x: torch.Tensor) -> torch.Tensor:
        """Compute emission scores from input IDs.

        Args:
            x: LongTensor [batch, seq_len]

        Returns:
            Tensor [batch, seq_len, num_tags]
        """
        emb = self.embedding(x)                  # [B, T, embed_dim]
        lstm_out, _ = self.lstm(emb)             # [B, T, hidden*2]
        emissions = self.output_proj(lstm_out)   # [B, T, num_tags]
        return emissions

    # -----------------------------------------------------------------
    # CRF: log partition (forward algorithm)
    # -----------------------------------------------------------------

    def _forward_alg(
        self, emissions: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        """Compute log-partition function (forward algorithm).

        Args:
            emissions: [batch, seq_len, num_tags]
            mask: [batch, seq_len] BoolTensor (True = valid)

        Returns:
            Scalar tensor (summed over batch).
        """
        batch_size, seq_len, num_tags = emissions.shape

        # alpha[b, t] = log-sum-exp of all paths ending in tag t at step 0
        alpha = emissions[:, 0, :]  # [B, num_tags]

        for t in range(1, seq_len):
            # alpha_expand: [B, num_tags, 1]  (previous tags)
            # trans:        [num_tags, num_tags]  trans[i,j] = j->i
            # emit:         [B, 1, num_tags]  (current tags)
            alpha_expand = alpha.unsqueeze(2)       # [B, num_tags, 1]
            trans = self.transitions.unsqueeze(0)   # [1, num_tags, num_tags]
            emit = emissions[:, t, :].unsqueeze(1)  # [B, 1, num_tags]

            scores = alpha_expand + trans + emit    # [B, num_tags, num_tags]
            new_alpha = torch.logsumexp(scores, dim=1)  # [B, num_tags]

            # Apply mask: only update positions that are valid
            m = mask[:, t].unsqueeze(1)  # [B, 1]
            alpha = new_alpha * m + alpha * (1 - m.float())

        # Total: logsumexp over final tags
        return torch.logsumexp(alpha, dim=1).sum()

    # -----------------------------------------------------------------
    # CRF: score of gold path
    # -----------------------------------------------------------------

    def _score_sentence(
        self,
        emissions: torch.Tensor,
        tags: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Score the gold tag sequence.

        Args:
            emissions: [batch, seq_len, num_tags]
            tags: [batch, seq_len] LongTensor
            mask: [batch, seq_len] BoolTensor

        Returns:
            Scalar tensor (summed over batch).
        """
        batch_size, seq_len, num_tags = emissions.shape

        # Emission scores for gold tags
        # Gather: emissions[b, t, tags[b,t]]
        emit_scores = emissions.gather(2, tags.unsqueeze(2)).squeeze(2)  # [B, T]
        emit_scores = (emit_scores * mask.float()).sum()

        # Transition scores
        trans_score = torch.tensor(0.0, device=emissions.device)
        for t in range(1, seq_len):
            m = mask[:, t].float()  # [B]
            # transitions[tags[b,t], tags[b,t-1]]
            cur = tags[:, t]        # [B]
            prev = tags[:, t - 1]   # [B]
            ts = self.transitions[cur, prev]  # [B]
            trans_score = trans_score + (ts * m).sum()

        return emit_scores + trans_score

    # -----------------------------------------------------------------
    # CRF: negative log-likelihood loss
    # -----------------------------------------------------------------

    def crf_loss(
        self,
        emissions: torch.Tensor,
        tags: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Negative log-likelihood CRF loss.

        Args:
            emissions: [batch, seq_len, num_tags]
            tags: [batch, seq_len] LongTensor
            mask: [batch, seq_len] BoolTensor

        Returns:
            Scalar loss tensor.
        """
        forward_score = self._forward_alg(emissions, mask)
        gold_score = self._score_sentence(emissions, tags, mask)
        return forward_score - gold_score

    # -----------------------------------------------------------------
    # CRF: Viterbi decode
    # -----------------------------------------------------------------

    def decode(
        self, emissions: torch.Tensor, mask: torch.Tensor
    ) -> List[List[int]]:
        """Viterbi decode to find best tag sequences.

        Args:
            emissions: [batch, seq_len, num_tags]
            mask: [batch, seq_len] BoolTensor

        Returns:
            List of tag ID lists, one per batch element.
        """
        batch_size, seq_len, num_tags = emissions.shape

        # Viterbi forward
        viterbi = emissions[:, 0, :].clone()  # [B, num_tags]
        backpointers: List[torch.Tensor] = []

        for t in range(1, seq_len):
            viterbi_expand = viterbi.unsqueeze(2)     # [B, num_tags, 1]
            trans = self.transitions.unsqueeze(0)      # [1, num_tags, num_tags]
            scores = viterbi_expand + trans             # [B, num_tags, num_tags]
            best_scores, best_tags = scores.max(dim=1) # [B, num_tags]
            emit = emissions[:, t, :]                   # [B, num_tags]
            new_viterbi = best_scores + emit

            m = mask[:, t].unsqueeze(1)
            viterbi = new_viterbi * m + viterbi * (1 - m.float())
            backpointers.append(best_tags)

        # Determine sequence lengths from mask
        lengths = mask.long().sum(dim=1)  # [B]

        # Backtrack
        results: List[List[int]] = []
        for b in range(batch_size):
            seq_end = lengths[b].item() - 1
            # Best last tag
            best_tag = viterbi[b].argmax().item()
            path = [best_tag]
            for t in range(len(backpointers) - 1, -1, -1):
                if t >= seq_end:
                    continue
                best_tag = backpointers[t][b, best_tag].item()
                path.append(best_tag)
            path.reverse()
            results.append(path[:lengths[b].item()])

        return results
