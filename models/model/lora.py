import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math


class LoRALayer(nn.Module):
    def __init__(self, input_dim, output_dim, rank, h=1):
        super(LoRALayer, self).__init__()
        if input_dim % h != 0:
            raise ValueError(f"input_dim must be divisible by h. Received input_dim={input_dim}, h={h}")
        self.rank = rank
        self.h = h
        self.input_dim = input_dim // h
        self.output_dim = output_dim // h

        self.A = nn.Parameter(torch.Tensor(self.input_dim, rank))
        self.B = nn.Parameter(torch.Tensor(rank, self.output_dim))
        self.register_parameter('W', None)

        self.init_parameters()

    def init_parameters(self):
        stdv = 1. / math.sqrt(self.A.size(1))
        self.A.data.uniform_(-stdv, stdv)
        self.B.data.uniform_(-stdv, stdv)

    def forward(self, x):
        if x.dim() == 3:  # If the input x has 3 dimensions, assuming it's (batch_size, seq_len, input_dim)
            batch_size, seq_len, _ = x.size()
            x = x.view(batch_size, seq_len, self.h, self.input_dim // self.h)
        elif x.dim() == 4:  # If the input x has 4 dimensions, assuming it's (batch_size, seq_len, h, input_dim)
            batch_size, seq_len, h, _ = x.size()
            assert h == self.h, "Input tensor does not match number of heads (h)"
        else:
            raise ValueError("Unsupported input dimensions")

        # Apply LoRA to the input x
        x = x.view(-1, self.input_dim)
        lora_output = x.matmul(self.A).matmul(self.B)
        lora_output = lora_output.view(batch_size, seq_len, self.h * self.output_dim)

        return lora_output


class SelfAttention(nn.Module):
    def __init__(self, d_model, d_k, d_v, h, attn_pdrop=0.2, resid_pdrop=0.2, lora_rank=2):
        super(SelfAttention, self).__init__()
        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v
        self.h = h

        # 使用 LoRALayer 替换原始的 Linear 层
        self.que_proj = LoRALayer(d_model, h * d_k, lora_rank, h)
        self.key_proj = LoRALayer(d_model, h * d_k, lora_rank, h)
        self.val_proj = LoRALayer(d_model, h * d_v, lora_rank, h)
        self.out_proj = nn.Linear(h * d_v, d_model)

        self.attn_drop = nn.Dropout(attn_pdrop)
        self.resid_drop = nn.Dropout(resid_pdrop)

    def forward(self, x, attention_mask=None):
        b_s, nq, _ = x.shape
        x=x.view(b_s,nq,self.h,self.d_model // self.h)

        q = self.que_proj(x)
        k = self.key_proj(x)
        v = self.val_proj(x)

        # 将q, k, v的形状变为 (batch_size, n_heads, seq_len, head_dim)
        q = q.view(b_s, nq, self.h, self.d_k).permute(0, 2, 1, 3)
        k = k.view(b_s, nq, self.h, self.d_k).permute(0, 2, 1, 3)
        v = v.view(b_s, nq, self.h, self.d_v).permute(0, 2, 1, 3)

        # 计算注意力分数
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        if attention_mask is not None:
            scores = scores.masked_fill(attention_mask == 0, float('-inf'))
        att = F.softmax(scores, dim=-1)
        att = self.attn_drop(att)

        # 应用注意力权重并合并头
        out = torch.matmul(att, v).transpose(1, 2).contiguous().view(b_s, nq, self.h * self.d_v)
        out = self.resid_drop(self.out_proj(out))

        return out


class myTransformerBlock(nn.Module):
    """ Transformer block """

    def __init__(self, d_model, d_k, d_v, h, block_exp, attn_pdrop, resid_pdrop,lora_rank):
        """
        :param d_model: Output dimensionality of the model
        :param d_k: Dimensionality of queries and keys
        :param d_v: Dimensionality of values
        :param h: Number of heads
        :param block_exp: Expansion factor for MLP (feed foreword network)

        """
        super().__init__()
        self.ln_input = nn.LayerNorm(d_model)
        self.ln_output = nn.LayerNorm(d_model)
        self.sa = SelfAttention(d_model, d_k, d_v, h, attn_pdrop, resid_pdrop, lora_rank)
        self.mlp = nn.Sequential(
            LoRALayer(d_model, block_exp * d_model, lora_rank, 1),  # Updated to use LoRALayer
            nn.GELU(),
            LoRALayer(block_exp * d_model, d_model, lora_rank, 1),  # Updated to use LoRALayer
            nn.Dropout(resid_pdrop),
        )
        

    def forward(self, x):
        bs, nx, c = x.size()

        x = x + self.sa(self.ln_input(x))
        x = x + self.mlp(self.ln_output(x))

        return x


class GPT_LORA(nn.Module):
    """  the full GPT language model, with a context size of block_size """

    def __init__(self, d_model, h=8, block_exp=4,
                 n_layer=8, vert_anchors=8, horz_anchors=8,
                 embd_pdrop=0.2, attn_pdrop=0.2, resid_pdrop=0.2 ,lora_rank=2):
        super().__init__()

        self.n_embd = d_model
        self.vert_anchors = vert_anchors
        self.horz_anchors = horz_anchors

        d_k = d_model//h
        d_v = d_model//h

        # positional embedding parameter (learnable), rgb_fea + ir_fea
        self.pos_emb = nn.Parameter(torch.zeros(1, 2 * vert_anchors * horz_anchors, self.n_embd))

        # transformer
        self.trans_blocks = nn.Sequential(*[
            myTransformerBlock(d_model, d_k, d_v, h, block_exp, attn_pdrop, resid_pdrop, lora_rank)
            for layer in range(n_layer)])

        # decoder head
        self.ln_f = nn.LayerNorm(self.n_embd)

        # regularization
        self.drop = nn.Dropout(embd_pdrop)

        # avgpool
        self.avgpool = nn.AdaptiveAvgPool2d((self.vert_anchors, self.horz_anchors))

        # init weights
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, x):
        """
        Args:
            x (tuple?)

        """
        rgb_fea = x[0]  # rgb_fea (tensor): dim:(B, C, H, W)
        ir_fea = x[1]   # ir_fea (tensor): dim:(B, C, H, W)
        assert rgb_fea.shape[0] == ir_fea.shape[0]
        bs, c, h, w = rgb_fea.shape

        # -------------------------------------------------------------------------
        # AvgPooling
        # -------------------------------------------------------------------------
        # AvgPooling for reduce the dimension due to expensive computation
        rgb_fea = self.avgpool(rgb_fea)
        ir_fea = self.avgpool(ir_fea)

        # -------------------------------------------------------------------------
        # Transformer
        # -------------------------------------------------------------------------
        # pad token embeddings along number of tokens dimension
        rgb_fea_flat = rgb_fea.view(bs, c, -1)  # flatten the feature
        ir_fea_flat = ir_fea.view(bs, c, -1)  # flatten the feature
        token_embeddings = torch.cat([rgb_fea_flat, ir_fea_flat], dim=2)  # concat
        token_embeddings = token_embeddings.permute(0, 2, 1).contiguous()  # dim:(B, 2*H*W, C)

        # transformer
        x = self.drop(self.pos_emb + token_embeddings)  # sum positional embedding and token    dim:(B, 2*H*W, C)
        x = self.trans_blocks(x)  # dim:(B, 2*H*W, C)

        # decoder head
        x = self.ln_f(x)  # dim:(B, 2*H*W, C)
        x = x.view(bs, 2, self.vert_anchors, self.horz_anchors, self.n_embd)
        x = x.permute(0, 1, 4, 2, 3)  # dim:(B, 2, C, H, W)

        # 这样截取的方式, 是否采用映射的方式更加合理？
        rgb_fea_out = x[:, 0, :, :, :].contiguous().view(bs, self.n_embd, self.vert_anchors, self.horz_anchors)
        ir_fea_out = x[:, 1, :, :, :].contiguous().view(bs, self.n_embd, self.vert_anchors, self.horz_anchors)

        # -------------------------------------------------------------------------
        # Interpolate (or Upsample)
        # -------------------------------------------------------------------------
        rgb_fea_out = F.interpolate(rgb_fea_out, size=([h, w]), mode='bilinear')
        ir_fea_out = F.interpolate(ir_fea_out, size=([h, w]), mode='bilinear')

        return rgb_fea_out, ir_fea_out
