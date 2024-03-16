import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init, Sequential
import numpy as np
import math

class LoRALayer(nn.Module):
    def __init__(self, in_features, out_features, rank, alpha=1.0):
        super(LoRALayer, self).__init__()
        self.rank = rank
        self.alpha = alpha

        # Original weight and bias
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.bias = nn.Parameter(torch.Tensor(out_features))

        # Low-rank matrices
        self.delta_W = nn.Parameter(torch.Tensor(rank, in_features))
        self.delta_B = nn.Parameter(torch.Tensor(out_features, rank))

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        nn.init.zeros_(self.bias)
        nn.init.normal_(self.delta_W, std=0.01)
        nn.init.normal_(self.delta_B, std=0.01)

    def forward(self, x):
        W = self.weight + self.alpha * (self.delta_B @ self.delta_W)
        return F.linear(x, W, self.bias)



class SelfAttention(nn.Module):
    """
    Multi-head masked self-attention layer with optional parameter sharing.
    """
    def __init__(self, d_model, h, attn_pdrop=0.2, resid_pdrop=0.2, shared_que_proj=None, shared_key_proj=None, shared_val_proj=None, shared_out_proj=None,rank=5):
        super(SelfAttention, self).__init__()
        self.d_model = d_model
        self.d_k = d_model // h
        self.d_v = d_model // h
        self.h = h

        self.que_proj = shared_que_proj if shared_que_proj is not None else LoRALayer(d_model, h * self.d_k, rank)
        self.key_proj = shared_key_proj if shared_key_proj is not None else LoRALayer(d_model, h * self.d_k, rank)
        self.val_proj = shared_val_proj if shared_val_proj is not None else LoRALayer(d_model, h * self.d_v, rank)
        self.out_proj = shared_out_proj if shared_out_proj is not None else LoRALayer(h * self.d_v, d_model, rank)


        self.attn_drop = nn.Dropout(attn_pdrop)
        self.resid_drop = nn.Dropout(resid_pdrop)

    def forward(self, x, attention_mask=None):
        b_s, nq = x.shape[:2]
        q = self.que_proj(x).view(b_s, nq, self.h, self.d_k).permute(0, 2, 1, 3)
        k = self.key_proj(x).view(b_s, nq, self.h, self.d_k).permute(0, 2, 3, 1)
        v = self.val_proj(x).view(b_s, nq, self.h, self.d_v).permute(0, 2, 1, 3)

        att = torch.matmul(q, k) / np.sqrt(self.d_k)
        if attention_mask is not None:
            att = att.masked_fill(attention_mask == 0, float('-inf'))
        att = torch.softmax(att, dim=-1)
        att = self.attn_drop(att)

        out = torch.matmul(att, v).permute(0, 2, 1, 3).contiguous().view(b_s, nq, self.h * self.d_v)
        out = self.resid_drop(out)
        out = self.out_proj(out)
        return out

class myTransformerBlock(nn.Module):
    """
    Transformer block that encapsulates a self-attention layer and a feed-forward network.
    """
    def __init__(self, d_model, h, block_exp, attn_pdrop, resid_pdrop, shared_que_proj=None, shared_key_proj=None, shared_val_proj=None, shared_out_proj=None,rank=5):
        super(myTransformerBlock, self).__init__()
        self.ln_input = nn.LayerNorm(d_model)
        self.ln_output = nn.LayerNorm(d_model)
        self.sa = SelfAttention(d_model, h, attn_pdrop, resid_pdrop, shared_que_proj, shared_key_proj, shared_val_proj, shared_out_proj)
        self.mlp = nn.Sequential(
            LoRALayer(d_model, block_exp * d_model,rank),
            nn.GELU(),
            nn.Dropout(resid_pdrop),
            LoRALayer(block_exp * d_model, d_model,rank),
            nn.Dropout(resid_pdrop),
        )

    def forward(self, x):
        x = x + self.sa(self.ln_input(x))
        x = x + self.mlp(self.ln_output(x))
        return x

class SHARED_GPT_LORA(nn.Module):
    """
    Full GPT model with shared projection matrices.
    """
    def __init__(self, d_model, h=8, block_exp=4, n_layer=8, vert_anchors=8, horz_anchors=8, embd_pdrop=0.2, attn_pdrop=0.2, resid_pdrop=0.2,rank=5):
        super().__init__()

        self.n_embd = d_model
        self.vert_anchors = vert_anchors
        self.horz_anchors = horz_anchors
        self.pos_emb = nn.Parameter(torch.zeros(1, 2 * vert_anchors * horz_anchors, self.n_embd))
        self.drop = nn.Dropout(embd_pdrop)


        # 创建共享的投影矩阵
        shared_que_proj = LoRALayer(d_model, h * (d_model // h), rank)
        shared_key_proj = LoRALayer(d_model, h * (d_model // h), rank)
        shared_val_proj = LoRALayer(d_model, h * (d_model // h), rank)
        shared_out_proj = LoRALayer(h * (d_model // h), d_model, rank)


        self.trans_blocks = nn.ModuleList([
            myTransformerBlock(d_model, h, block_exp, attn_pdrop, resid_pdrop, shared_que_proj, shared_key_proj, shared_val_proj, shared_out_proj)
            for _ in range(n_layer)])


        self.ln_f = nn.LayerNorm(self.n_embd)
        self.avgpool = nn.AdaptiveAvgPool2d((vert_anchors, horz_anchors))

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

    # Forward method remains unchanged...

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
        # Pass through each Transformer block in the ModuleList
        for block in self.trans_blocks:
            x = block(x)

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
    