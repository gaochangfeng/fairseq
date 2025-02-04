import torch

from torch import nn

from .layer_norm import LayerNorm


class DecoderLayer(nn.Module):
    """Single decoder layer module

    :param int size: input dim
    :param espnet.nets.pytorch_backend.transformer.attention.MultiHeadedAttention self_attn: self attention module
    :param espnet.nets.pytorch_backend.transformer.attention.MultiHeadedAttention src_attn: source attention module
    :param espnet.nets.pytorch_backend.transformer.positionwise_feed_forward.PositionwiseFeedForward feed_forward:
        feed forward layer module
    :param float dropout_rate: dropout rate
    :param bool normalize_before: whether to use layer_norm before the first block
    :param bool concat_after: whether to concat attention layer's input and output
        if True, additional linear will be applied. i.e. x -> x + linear(concat(x, att(x)))
        if False, no additional linear will be applied. i.e. x -> x + att(x)
    """

    def __init__(self, size, self_attn, src_attn, feed_forward, dropout_rate,
                 normalize_before=True, concat_after=False, domain_dim=0):
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.norm1 = LayerNorm(size)
        self.norm2 = LayerNorm(size)
        self.norm3 = LayerNorm(size)
        self.dropout = nn.Dropout(dropout_rate)
        self.normalize_before = normalize_before
        self.concat_after = concat_after
        if self.concat_after:
            self.concat_linear1 = nn.Linear(size + size + domain_dim, size)
            self.concat_linear2 = nn.Linear(size + size + domain_dim, size)

    def forward(self, tgt, tgt_mask, memory, memory_mask, tags, cache=None, ep=None):
        """Compute decoded features.

        Args:
            tgt (torch.Tensor): Input tensor (#batch, maxlen_out, size).
            tgt_mask (torch.Tensor): Mask for input tensor (#batch, maxlen_out).
            memory (torch.Tensor): Encoded memory, float32 (#batch, maxlen_in, size).
            memory_mask (torch.Tensor): Encoded memory mask (#batch, maxlen_in).
            tags (torch.Tensor): domain tag (batch, maxlen_out, domain_dim).
            cache (List[torch.Tensor]): List of cached tensors.
                Each tensor shape should be (#batch, maxlen_out - 1, size).
            ep (List[int]): List of end pointers for streaming decoder.

        Returns:
            torch.Tensor: Output tensor(#batch, maxlen_out, size).
            torch.Tensor: Mask for output tensor (#batch, maxlen_out).
            torch.Tensor: Encoded memory (#batch, maxlen_in, size).
            torch.Tensor: Encoded memory mask (#batch, maxlen_in).

        """
        residual = tgt
        if self.normalize_before:
            tgt = self.norm1(tgt)
        if tags is not None:
            tgt = torch.cat((tgt, tags), dim=-1)
        if cache is None:
            tgt_q = tgt
            tgt_q_mask = tgt_mask
            tgt_q_tags = tags
        else:
            # compute only the last frame query keeping dim: max_time_out -> 1
            assert cache.shape == (
                tgt.shape[0],
                tgt.shape[1] - 1,
                self.size,
            ), f"{cache.shape} == {(tgt.shape[0], tgt.shape[1] - 1, self.size)}"
            tgt_q = tgt[:, -1:, :]
            residual = residual[:, -1:, :]
            tgt_q_mask = None
            if tgt_mask is not None:
                tgt_q_mask = tgt_mask[:, -1:, :]
            tgt_q_tags = None
            if tags is not None:
                tgt_q_tags = tags[:, -1:, :]

        if self.concat_after:
            if tgt_q_tags is None:
                tgt_concat = torch.cat(
                    (tgt_q, self.self_attn(tgt_q, tgt, tgt, tgt_q_mask, tgt_q_tags)), dim=-1
                )
            else:
                tgt_concat = torch.cat(
                    (tgt_q, self.self_attn(tgt_q, tgt, tgt, tgt_q_mask, tgt_q_tags), tgt_q_tags), dim=-1
                )
            x = residual + self.concat_linear1(tgt_concat)
        else:
            x = residual + self.dropout(self.self_attn(tgt_q, tgt, tgt, tgt_q_mask, tgt_q_tags))
        if not self.normalize_before:
            x = self.norm1(x)

        residual = x
        if self.normalize_before:
            x = self.norm2(x)
        if tgt_q_tags is not None:
            x = torch.cat((x, tgt_q_tags), dim=-1)
        if self.concat_after:
            if tgt_q_tags is None:
                if ep is None:
                    x_concat = torch.cat(
                        (x, self.src_attn(x, memory, memory, memory_mask, tgt_q_tags)), dim=-1
                    )
                else:
                    x_concat = torch.cat(
                        (x, self.src_attn(x, memory, memory, memory_mask, tgt_q_tags, ep)), dim=-1
                    )
            else:
                if ep is None:
                    x_concat = torch.cat(
                        (x, self.src_attn(x, memory, memory, memory_mask, tgt_q_tags), tgt_q_tags), dim=-1
                    )
                else:
                    x_concat = torch.cat(
                        (x, self.src_attn(x, memory, memory, memory_mask, tgt_q_tags, ep), tgt_q_tags), dim=-1
                    )
            x = residual + self.concat_linear2(x_concat)
        else:
            if ep is None:
                x = residual + self.dropout(self.src_attn(x, memory, memory, memory_mask, tgt_q_tags))
            else:
                x = residual + self.dropout(self.src_attn(x, memory, memory, memory_mask, tgt_q_tags, ep))
        if not self.normalize_before:
            x = self.norm2(x)

        residual = x
        if self.normalize_before:
            x = self.norm3(x)
        x = residual + self.dropout(self.feed_forward(x, tgt_q_tags))
        if not self.normalize_before:
            x = self.norm3(x)

        if cache is not None:
            x = torch.cat([cache, x], dim=1)

        return x, tgt_mask, memory, memory_mask, tags
