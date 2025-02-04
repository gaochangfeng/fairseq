import torch

from .attention import MultiHeadedAttention
from .attention import MTMultiHeadedAttention
from .decoder_layer import DecoderLayer
from .embedding import PositionalEncoding
from .layer_norm import LayerNorm
from .positionwise_feed_forward import PositionwiseFeedForward
from .repeat import repeat


class Decoder(torch.nn.Module):
    """Transfomer decoder module

    :param int odim: output dim
    :param int attention_dim: dimention of attention
    :param int attention_heads: the number of heads of multi head attention
    :param int linear_units: the number of units of position-wise feed forward
    :param int num_blocks: the number of decoder blocks
    :param float dropout_rate: dropout rate
    :param float attention_dropout_rate: dropout rate for attention
    :param str or torch.nn.Module input_layer: input layer type
    :param bool use_output_layer: whether to use output layer
    :param class pos_enc_class: PositionalEncoding or ScaledPositionalEncoding
    :param bool normalize_before: whether to use layer_norm before the first block
    :param bool concat_after: whether to concat attention layer's input and output
        if True, additional linear will be applied. i.e. x -> x + linear(concat(x, att(x)))
        if False, no additional linear will be applied. i.e. x -> x + att(x)
    """

    def __init__(self, odim,
                 attention_dim=256,
                 attention_heads=4,
                 linear_units=2048,
                 num_blocks=6,
                 dropout_rate=0.1,
                 positional_dropout_rate=0.1,
                 self_attention_dropout_rate=0.0,
                 src_attention_dropout_rate=0.0,
                 input_layer="embed",
                 use_output_layer=True,
                 pos_enc=None,
                 active_func=torch.nn.ReLU(),
                 normalize_before=True,
                 concat_after=False,
                 domain_dim=0):
        super(Decoder, self).__init__()
        if pos_enc is None:
            pos_enc = PositionalEncoding(attention_dim, positional_dropout_rate)
        if input_layer == "embed":
            self.embed = torch.nn.Sequential(
                torch.nn.Embedding(odim, attention_dim),
                pos_enc
            )
        elif input_layer == "linear":
            self.embed = torch.nn.Sequential(
                torch.nn.Linear(odim, attention_dim),
                torch.nn.LayerNorm(attention_dim),
                torch.nn.Dropout(dropout_rate),
                torch.nn.ReLU(),
                pos_enc
            )
        elif isinstance(input_layer, torch.nn.Module):
            self.embed = torch.nn.Sequential(
                input_layer,
                pos_enc
            )
        else:
            raise NotImplementedError("only `embed` or torch.nn.Module is supported.")
        self.normalize_before = normalize_before
        self.decoders = repeat(
            num_blocks,
            lambda lnum: DecoderLayer(
                attention_dim,
                MultiHeadedAttention(attention_heads, attention_dim, self_attention_dropout_rate, domain_dim),
                MultiHeadedAttention(attention_heads, attention_dim, src_attention_dropout_rate, domain_dim),
                PositionwiseFeedForward(attention_dim, linear_units, dropout_rate, domain_dim, active_func),
                dropout_rate,
                normalize_before,
                concat_after,
                domain_dim
            )
        )
        if self.normalize_before:
            self.after_norm = LayerNorm(attention_dim)
        if use_output_layer:
            self.output_layer = torch.nn.Linear(attention_dim + domain_dim, odim)
        else:
            self.output_layer = None

    def forward(self, tgt, tgt_mask, memory, memory_mask, tags=None):
        """forward decoder

        :param torch.Tensor tgt: input token ids, int64 (batch, maxlen_out) if input_layer == "embed"
                                 input tensor (batch, maxlen_out, #mels) in the other cases
        :param torch.Tensor tgt_mask: input token mask, uint8  (batch, maxlen_out)
        :param torch.Tensor memory: encoded memory, float32  (batch, maxlen_in, feat)
        :param torch.Tensor memory_mask: encoded memory mask, uint8  (batch, maxlen_in)
        :param torch.Tensor tags: domain id, float32  (batch, domain)
        :return x: decoded token score before softmax (batch, maxlen_out, token) if use_output_layer is True,
                   final block outputs (batch, maxlen_out, attention_dim) in the other cases
        :rtype: torch.Tensor
        :return tgt_mask: score mask before softmax (batch, maxlen_out)
        :rtype: torch.Tensor
        """
        x = self.embed(tgt)
        if tags is not None:
            tags = tags.unsqueeze(1).repeat(1,x.size(1),1)
        x, tgt_mask, memory, memory_mask, _ = self.decoders(x, tgt_mask, memory, memory_mask, tags)
        if self.normalize_before:
            x = self.after_norm(x)
        if self.output_layer is not None:
            if tags is not None:
                x = torch.cat((x, tags), dim=-1)
            x = self.output_layer(x)
        return x, tgt_mask

    def forward_one_step(self, tgt, tgt_mask, memory, cache=None, tags=None):
        """Forward one step.

        Args:
            tgt (torch.Tensor): Input token ids, int64 (#batch, maxlen_out).
            tgt_mask (torch.Tensor): Input token mask (#batch, maxlen_out).
                dtype=torch.uint8 in PyTorch 1.2- and dtype=torch.bool in PyTorch 1.2+
                (include 1.2).
            memory (torch.Tensor): Encoded memory, float32 (#batch, maxlen_in, feat).
            cache (List[torch.Tensor]): List of cached tensors.
                Each tensor shape should be (#batch, maxlen_out - 1, size).

        Returns:
            torch.Tensor: Output tensor (batch, maxlen_out, odim).
            List[torch.Tensor]: List of cache tensors of each decoder layer.

        """
        x = self.embed(tgt)
        if tags is not None:
            tags = tags.unsqueeze(1).repeat(1,x.size(1),1)
        if cache is None:
            cache = [None] * len(self.decoders)
        new_cache = []
        for c, decoder in zip(cache, self.decoders):
            x, tgt_mask, memory, memory_mask, _ = decoder(
                x, tgt_mask, memory, None, tags, cache=c
            )
            new_cache.append(x)

        if self.normalize_before:
            y = self.after_norm(x[:, -1])
        else:
            y = x[:, -1]
        if self.output_layer is not None:
            if tags is not None:
                y = torch.cat((y, tags[:, -1]), dim=-1)
            y = torch.log_softmax(self.output_layer(y), dim=-1)

        return y, new_cache

    def forward_one_step_batch(self, tgt, tgt_mask, memory, memory_mask, cache, tags=None):
        """Forward one step.
        :param torch.Tensor tgt: input token ids, int64 (batch, maxlen_out)
        :param torch.Tensor tgt_mask: input token mask,  (batch, maxlen_out)
                                      dtype=torch.uint8 in PyTorch 1.2-
                                      dtype=torch.bool in PyTorch 1.2+ (include 1.2)
        :param torch.Tensor memory: encoded memory, float32  (batch, maxlen_in, feat)
        :param List[torch.Tensor] cache:
            cached output list of (batch, max_time_out-1, size)
        :return y, cache: NN output value and cache per `self.decoders`.
            `y.shape` is (batch, maxlen_out, token)
        :rtype: Tuple[torch.Tensor, List[torch.Tensor]]
        """
        x = self.embed(tgt)
        if tags is not None:
            tags = tags.unsqueeze(1).repeat(1,x.size(1),1)
        new_cache = []
        for i, (c, decoder) in enumerate(zip(cache, self.decoders)):
            x, tgt_mask, memory, memory_mask = decoder(x, tgt_mask, memory, memory_mask, tags, cache=c)
            new_cache.append((x,decoder.attn) if i==2 else x)

        if self.normalize_before:
            self.output_embedding = self.after_norm(x)
            y = self.output_embedding[:, -1]
        else:
            y = x[:, -1]
            self.output_embedding = x
        if self.output_layer is not None:
            if tags is not None:
                y = torch.cat((y, tags[:, -1]), dim=-1)
            y = torch.log_softmax(self.output_layer(y), dim=-1)
        return y, new_cache 

    def recognize(self, tgt, tgt_mask, memory, tags=None):
        """recognize one step

        :param torch.Tensor tgt: input token ids, int64 (batch, maxlen_out)
        :param torch.Tensor tgt_mask: input token mask, uint8  (batch, maxlen_out)
        :param torch.Tensor memory: encoded memory, float32  (batch, maxlen_in, feat)
        :return x: decoded token score before softmax (batch, maxlen_out, token)
        :rtype: torch.Tensor
        """
        x = self.embed(tgt)
        if tags is not None:
            tags = tags.unsqueeze(1).repeat(1,x.size(1),1)
        x, tgt_mask, memory, memory_mask, _ = self.decoders(x, tgt_mask, memory, None, tags)
        if self.normalize_before:
            x_ = self.after_norm(x[:, -1])
        else:
            x_ = x[:, -1]
        if self.output_layer is not None:
            if tags is not None:
                x_ = torch.cat((x_, tags[:, -1]), dim=-1)
            return torch.log_softmax(self.output_layer(x_), dim=-1)
        else:
            return x_

class StreamDecoder(torch.nn.Module):
    """Transfomer decoder module for streaming

    :param int odim: output dim
    :param int attention_dim: dimention of attention
    :param int attention_heads: the number of heads of multi head attention
    :param int linear_units: the number of units of position-wise feed forward
    :param int num_blocks: the number of decoder blocks
    :param float dropout_rate: dropout rate
    :param float attention_dropout_rate: dropout rate for attention
    :param str or torch.nn.Module input_layer: input layer type
    :param bool use_output_layer: whether to use output layer
    :param class pos_enc_class: PositionalEncoding or ScaledPositionalEncoding
    :param bool normalize_before: whether to use layer_norm before the first block
    :param bool concat_after: whether to concat attention layer's input and output
        if True, additional linear will be applied. i.e. x -> x + linear(concat(x, att(x)))
        if False, no additional linear will be applied. i.e. x -> x + att(x)
    """

    def __init__(self, odim,
                 attention_dim=256,
                 self_attention_heads=4,
                 src_attention_heads=1,
                 linear_units=2048,
                 num_blocks=6,
                 dropout_rate=0.1,
                 positional_dropout_rate=0.1,
                 self_attention_dropout_rate=0.0,
                 src_attention_dropout_rate=0.0,
                 src_attention_bias_init=0.0,
                 src_attention_sigmoid_noise=1.0,
                 input_layer="embed",
                 use_output_layer=True,
                 pos_enc=None,
                 normalize_before=True,
                 concat_after=False,
                 domain_dim=0):
        super(StreamDecoder, self).__init__()
        if pos_enc is None:
            pos_enc = PositionalEncoding(attention_dim, positional_dropout_rate)
        if input_layer == "embed":
            self.embed = torch.nn.Sequential(
                torch.nn.Embedding(odim, attention_dim),
                pos_enc
            )
        elif input_layer == "linear":
            self.embed = torch.nn.Sequential(
                torch.nn.Linear(odim, attention_dim),
                torch.nn.LayerNorm(attention_dim),
                torch.nn.Dropout(dropout_rate),
                torch.nn.ReLU(),
                pos_enc
            )
        elif isinstance(input_layer, torch.nn.Module):
            self.embed = torch.nn.Sequential(
                input_layer,
                pos_enc
            )
        else:
            raise NotImplementedError("only `embed` or torch.nn.Module is supported.")
        self.normalize_before = normalize_before
        self.decoders = repeat(
            num_blocks,
            lambda lnum: DecoderLayer(
                attention_dim,
                MultiHeadedAttention(self_attention_heads, attention_dim, self_attention_dropout_rate, domain_dim),
                MTMultiHeadedAttention(src_attention_heads, attention_dim, src_attention_dropout_rate, \
                                       src_attention_bias_init, src_attention_sigmoid_noise, domain_dim),
                PositionwiseFeedForward(attention_dim, linear_units, dropout_rate, domain_dim),
                dropout_rate,
                normalize_before,
                concat_after,
                domain_dim,
            )
        )
        if self.normalize_before:
            self.after_norm = LayerNorm(attention_dim)
        if use_output_layer:
            self.output_layer = torch.nn.Linear(attention_dim + domain_dim, odim)
        else:
            self.output_layer = None

    def forward(self, tgt, tgt_mask, memory, memory_mask, tags=None):
        """forward decoder

        :param torch.Tensor tgt: input token ids, int64 (batch, maxlen_out) if input_layer == "embed"
                                 input tensor (batch, maxlen_out, #mels) in the other cases
        :param torch.Tensor tgt_mask: input token mask, uint8  (batch, maxlen_out)
        :param torch.Tensor memory: encoded memory, float32  (batch, maxlen_in, feat)
        :param torch.Tensor memory_mask: encoded memory mask, uint8  (batch, maxlen_in)
        :return x: decoded token score before softmax (batch, maxlen_out, token) if use_output_layer is True,
                   final block outputs (batch, maxlen_out, attention_dim) in the other cases
        :rtype: torch.Tensor
        :return tgt_mask: score mask before softmax (batch, maxlen_out)
        :rtype: torch.Tensor
        """
        x = self.embed(tgt)
        if tags is not None:
            tags = tags.unsqueeze(1).repeat(1,x.size(1),1)
        x, tgt_mask, memory, memory_mask, _ = self.decoders(x, tgt_mask, memory, memory_mask, tags)
        if self.normalize_before:
            x = self.after_norm(x)
        if self.output_layer is not None:
            if tags is not None:
                x = torch.cat((x, tags), dim=-1)
            x = self.output_layer(x)
        return x, tgt_mask

    def forward_one_step_online(self, tgt, tgt_mask, memory, cache=None, tags=None):
        """Forward one step.
        :param torch.Tensor tgt: input token ids, int64 (batch, maxlen_out)
        :param torch.Tensor tgt_mask: input token mask,  (batch, maxlen_out)
                                      dtype=torch.uint8 in PyTorch 1.2-
                                      dtype=torch.bool in PyTorch 1.2+ (include 1.2)
        :param torch.Tensor memory: encoded memory, float32  (batch, maxlen_in, feat)
        :param List[torch.Tensor] cache:
            cached output list of (batch, max_time_out-1, size)
        :return y, cache: NN output value and cache per `self.decoders`.
            `y.shape` is (batch, maxlen_out, token)
        :rtype: Tuple[torch.Tensor, List[torch.Tensor]]
        """
        x = self.embed(tgt)
        if tags is not None:
            tags = tags.unsqueeze(1).repeat(1,x.size(1),1)
        if cache is None:
            cache = [(None, -1) for _ in range(len(self.decoders))]
        new_cache = []
        for c, decoder in zip(cache, self.decoders):
            x, tgt_mask, memory, memory_mask, _ = decoder(x, tgt_mask, memory, None, tags, cache=c[0], ep=c[1])
            new_cache.append((x, [k for k in decoder.src_attn.endpoint]))

        if self.normalize_before:
            y = self.after_norm(x[:, -1])
        else:
            y = x[:, -1]
        if self.output_layer is not None:
            if tags is not None:
                y = torch.cat((y, tags[:, -1]), dim=-1)
            y = torch.log_softmax(self.output_layer(y), dim=-1)

        return y, new_cache

    def forward_one_step_offline(self, tgt, tgt_mask, memory, cache=None, tags=None):
        """Forward one step.
        :param torch.Tensor tgt: input token ids, int64 (batch, maxlen_out)
        :param torch.Tensor tgt_mask: input token mask,  (batch, maxlen_out)
                                      dtype=torch.uint8 in PyTorch 1.2-
                                      dtype=torch.bool in PyTorch 1.2+ (include 1.2)
        :param torch.Tensor memory: encoded memory, float32  (batch, maxlen_in, feat)
        :param List[torch.Tensor] cache:
            cached output list of (batch, max_time_out-1, size)
        :return y, cache: NN output value and cache per `self.decoders`.
            `y.shape` is (batch, maxlen_out, token)
        :rtype: Tuple[torch.Tensor, List[torch.Tensor]]
        """
        x = self.embed(tgt)
        if tags is not None:
            tags = tags.unsqueeze(1).repeat(1,x.size(1),1)
        if cache is None:
            cache = [None for _ in range(len(self.decoders))]
        new_cache = []
        for c, decoder in zip(cache, self.decoders):
            x, tgt_mask, memory, memory_mask, _ = decoder(x, tgt_mask, memory, None, tags, cache=c)
            new_cache.append(x)

        if self.normalize_before:
            y = self.after_norm(x[:, -1])
        else:
            y = x[:, -1]
        if self.output_layer is not None:
            if tags is not None:
                y = torch.cat((y, tags[:, -1]), dim=-1)
            y = torch.log_softmax(self.output_layer(y), dim=-1)

        return y, new_cache
    
    def forward_one_step(self, tgt, tgt_mask, memory, cache=None, tags=None, online=False):
        if online:
            return self.forward_one_step_online(tgt, tgt_mask, memory, cache=cache, tags=tags)
        else:
            return self.forward_one_step_offline(tgt, tgt_mask, memory, cache=cache, tags=tags)
