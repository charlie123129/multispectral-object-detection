import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.nn import init, Sequential
import math

"""
class QLoRALayer1(nn.Module):
    def __init__(self, in_features, out_features, rank, alpha=1.0, n_bits=8):
        super(QLoRALayer1, self).__init__()
        self.rank = rank
        self.alpha = alpha
        self.n_bits = n_bits  # Number of bits for quantization

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

    def quantize(self, x, n_bits):
        # Simulate quantization process here
        qmin = -2 ** (n_bits - 1)
        qmax = 2 ** (n_bits - 1) - 1
        scale = (x.max() - x.min()) / (qmax - qmin)
        x_scaled = torch.round(x / scale) + qmin
        x_quantized = x_scaled * scale
        return x_quantized

    def forward(self, x):
        quantized_delta_W = self.quantize(self.delta_W, self.n_bits)
        quantized_delta_B = self.quantize(self.delta_B, self.n_bits)
        W = self.weight + self.alpha * (quantized_delta_B @ quantized_delta_W)
        return F.linear(x, W, self.bias)
"""

class QLoRALayer(nn.Module):
    def __init__(self, in_features, out_features, rank, alpha=1.0, n_bits=8):
        super(QLoRALayer, self).__init__()
        self.rank = rank
        self.alpha = alpha
        self.n_bits = n_bits  # Number of bits for quantization

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
        nn.init.normal_(self.delta_W, std=0.02)
        nn.init.normal_(self.delta_B, std=0.02)

    def quantize(self, x, n_bits):
        # Simulate quantization
        qmin = -2.0 ** (n_bits - 1)
        qmax = 2.0 ** (n_bits - 1) - 1
        min_val, max_val = x.min(), x.max()
        scale = (max_val - min_val) / (qmax - qmin)
        scale = max(scale, 1e-6)  # Prevent division by zero
        zero_point = qmin - min_val / scale
        quantized = torch.round(x / scale + zero_point)
        # Dequantize
        dequantized = (quantized - zero_point) * scale
        return dequantized

    def forward(self, x):
        # Quantize delta_W and delta_B
        quantized_delta_W = self.quantize(self.delta_W, self.n_bits)
        quantized_delta_B = self.quantize(self.delta_B, self.n_bits)

        W = self.weight + self.alpha * (quantized_delta_B @ quantized_delta_W)
        return F.linear(x, W, self.bias)


class SelfAttention(nn.Module):
    def __init__(self, d_model, d_k, d_v, h, attn_pdrop=0.2, resid_pdrop=0.2, lora_rank=32, lora_alpha=1.0, n_bits=8):
        super(SelfAttention, self).__init__()
        self.d_model = d_model
        self.d_k = d_model // h
        self.d_v = d_model // h
        self.h = h

        self.que_proj = QLoRALayer(d_model, h * self.d_k, lora_rank, lora_alpha, n_bits)
        self.key_proj = QLoRALayer(d_model, h * self.d_k, lora_rank, lora_alpha, n_bits)
        self.val_proj = QLoRALayer(d_model, h * self.d_v, lora_rank, lora_alpha, n_bits)
        self.out_proj = QLoRALayer(h * self.d_v, d_model, lora_rank, lora_alpha, n_bits)

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
        att = torch.softmax(att, -1)
        att = self.attn_drop(att)

        out = torch.matmul(att, v).permute(0, 2, 1, 3).contiguous().view(b_s, nq, self.h * self.d_v)
        out = self.resid_drop(self.out_proj(out))
        return out
    

class myTransformerBlock(nn.Module):
    def __init__(self, d_model, d_k, d_v, h, block_exp, attn_pdrop, resid_pdrop, lora_rank=32, lora_alpha=2.0, n_bits=8):
        super(myTransformerBlock, self).__init__()
        self.ln_input = nn.LayerNorm(d_model)
        self.ln_output = nn.LayerNorm(d_model)

        # SelfAttention 层现在使用 QLoRALayer
        self.sa = SelfAttention(d_model, d_k, d_v, h, attn_pdrop, resid_pdrop, lora_rank, lora_alpha, n_bits)
        
        # MLP层，使用QLoRALayer进行量化
        self.mlp = nn.Sequential(
            QLoRALayer(d_model, block_exp * d_model, lora_rank, lora_alpha, n_bits),
            nn.GELU(),
            QLoRALayer(block_exp * d_model, d_model, lora_rank, lora_alpha, n_bits),
            nn.Dropout(resid_pdrop),
        )

    def forward(self, x):
        x = x + self.sa(self.ln_input(x))
        x = x + self.mlp(self.ln_output(x))
        return x


class GPT_QLORA_ADJUST(nn.Module):
    def __init__(self, d_model, h=8, block_exp=4, n_layer=8, vert_anchors=8, horz_anchors=8,
                 embd_pdrop=0.2, attn_pdrop=0.2, resid_pdrop=0.2, lora_rank=32, lora_alpha=2.0, n_bits=8):
        super(GPT_QLORA_ADJUST, self).__init__()

        self.n_embd = d_model
        self.vert_anchors = vert_anchors
        self.horz_anchors = horz_anchors

        # Positional embedding parameter
        self.pos_emb = nn.Parameter(torch.zeros(1, 2 * vert_anchors * horz_anchors, self.n_embd))

        self.current_alpha = lora_alpha  # 先直接存储alpha值，不调用adjust_alpha

        # Transformer blocks using QLoRALayer
        self.trans_blocks = nn.Sequential(*[
            myTransformerBlock(d_model, d_model // h, d_model // h, h, block_exp, attn_pdrop, resid_pdrop, lora_rank, self.current_alpha, n_bits)
            for _ in range(n_layer)])
        
        self.adjust_alpha(lora_alpha)  # 在所有属性初始化之后调用adjust_alpha

        self.ln_f = nn.LayerNorm(self.n_embd)
        self.drop = nn.Dropout(embd_pdrop)
        self.avgpool = nn.AdaptiveAvgPool2d((self.vert_anchors, self.horz_anchors))

        self.apply(self._init_weights)

    def adjust_alpha(self, new_alpha):
        """
        动态调整LoRA层的alpha值。
        """
        self.current_alpha = new_alpha
        for block in self.trans_blocks:
            for module in block.modules():
                if isinstance(module, QLoRALayer):
                    module.alpha = self.current_alpha

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
        rgb_fea, ir_fea = x
        bs, c, h, w = rgb_fea.shape

        rgb_fea = self.avgpool(rgb_fea)
        ir_fea = self.avgpool(ir_fea)

        rgb_fea_flat = rgb_fea.view(bs, c, -1)
        ir_fea_flat = ir_fea.view(bs, c, -1)
        token_embeddings = torch.cat([rgb_fea_flat, ir_fea_flat], dim=2).permute(0, 2, 1)

        x = self.drop(self.pos_emb + token_embeddings)
        x = self.trans_blocks(x)

        x = self.ln_f(x)
        x = x.view(bs, 2, self.vert_anchors, self.horz_anchors, self.n_embd).permute(0, 1, 4, 2, 3)

        rgb_fea_out = x[:, 0].contiguous().view(bs, self.n_embd, self.vert_anchors, self.horz_anchors)
        ir_fea_out = x[:, 1].contiguous().view(bs, self.n_embd, self.vert_anchors, self.horz_anchors)

        rgb_fea_out = F.interpolate(rgb_fea_out, size=(h, w), mode='bilinear')
        ir_fea_out = F.interpolate(ir_fea_out, size=(h, w), mode='bilinear')

        return rgb_fea_out, ir_fea_out

