import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.nn import init, Sequential
import math


class FakeQuantize(nn.Module):
    """
    模拟量化操作，与之前的实现相同。
    """
    def __init__(self, n_bits=8):
        super(FakeQuantize, self).__init__()
        self.n_bits = n_bits
        self.scale = nn.Parameter(torch.Tensor([0.1]), requires_grad=False)
        self.zero_point = nn.Parameter(torch.Tensor([0]), requires_grad=False)

    def forward(self, x):
        qmin = -2 ** (self.n_bits - 1)
        qmax = 2 ** (self.n_bits - 1) - 1
        x = torch.round(x / self.scale + self.zero_point).clamp(qmin, qmax)
        x = (x - self.zero_point) * self.scale
        return x

class QALoRALayer(nn.Module):
    """
    根据伪代码修改后的量化感知LoRA层。
    """
    def __init__(self, in_features, out_features, rank, alpha=1.0, n_bits=8, L=1):
        super(QALoRALayer, self).__init__()
        self.rank = rank
        self.alpha = alpha
        self.L = L  # 新增分组量化的分组大小

        self.quantize = FakeQuantize(n_bits=n_bits)

        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.delta_W = nn.Parameter(torch.Tensor(rank, in_features // L))
        self.delta_B = nn.Parameter(torch.Tensor(out_features, rank))

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        nn.init.normal_(self.delta_W, std=0.01)
        nn.init.normal_(self.delta_B, std=0.01)

    def forward(self, x):
        # 分组平均池化操作，模拟QA操作
        x_avg = F.avg_pool1d(x.transpose(1, 2), kernel_size=self.L).transpose(1, 2)
        
        # 量化权重和低秩矩阵
        weight_q = self.quantize(self.weight)
        delta_W_q = self.quantize(self.delta_W)
        delta_B_q = self.quantize(self.delta_B)

        W_tilde = weight_q + self.alpha * torch.matmul(delta_B_q, delta_W_q)
        output = F.linear(x, W_tilde)

        # 应用低秩矩阵调整后的结果
        output += self.alpha * F.linear(x_avg, torch.matmul(delta_B_q, delta_W_q).transpose(0, 1))

        return output

class SelfAttention(nn.Module):
    def __init__(self, d_model, d_k, d_v, h, attn_pdrop=0.2, resid_pdrop=0.2, lora_rank=32, lora_alpha=2.0, lora_beta=0.0):
        super(SelfAttention, self).__init__()
        self.h = h
        
        # 注意：为适应QLoRALayer，我们需要调整参数
        self.que_proj = QALoRALayer(d_model, h * d_k, lora_rank, d_model // h, s=lora_alpha, alpha=lora_alpha, beta=lora_beta, W=None)
        self.key_proj = QALoRALayer(d_model, h * d_k, lora_rank, d_model // h, s=lora_alpha, alpha=lora_alpha, beta=lora_beta, W=None)
        self.val_proj = QALoRALayer(d_model, h * d_v, lora_rank, d_model // h, s=lora_alpha, alpha=lora_alpha, beta=lora_beta, W=None)
        self.out_proj = QALoRALayer(h * d_v, d_model, lora_rank, d_model // h, s=lora_alpha, alpha=lora_alpha, beta=lora_beta, W=None)

        self.attn_drop = nn.Dropout(attn_pdrop)
        self.resid_drop = nn.Dropout(resid_pdrop)


        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x, attention_mask=None, attention_weights=None):
        '''
        Computes Self-Attention
        Args:
            x (tensor): input (token) dim:(b_s, nx, c),
                b_s means batch size
                nx means length, for CNN, equals H*W, i.e. the length of feature maps
                c means channel, i.e. the channel of feature maps
            attention_mask: Mask over attention values (b_s, h, nq, nk). True indicates masking.
            attention_weights: Multiplicative weights for attention values (b_s, h, nq, nk).
        Return:
            output (tensor): dim:(b_s, nx, c)
        '''

        b_s, nq = x.shape[:2]
        nk = x.shape[1]
        q = self.que_proj(x).view(b_s, nq, self.h, self.d_k).permute(0, 2, 1, 3)  # (b_s, h, nq, d_k)
        k = self.key_proj(x).view(b_s, nk, self.h, self.d_k).permute(0, 2, 3, 1)  # (b_s, h, d_k, nk) K^T
        v = self.val_proj(x).view(b_s, nk, self.h, self.d_v).permute(0, 2, 1, 3)  # (b_s, h, nk, d_v)

        # Self-Attention
        #  :math:`(\text(Attention(Q,K,V) = Softmax((Q*K^T)/\sqrt(d_k))`
        att = torch.matmul(q, k) / np.sqrt(self.d_k)  # (b_s, h, nq, nk)

        # weight and mask
        #if attention_weights is not None:
        #    att = att * attention_weights
        #if attention_mask is not None:
        #    att = att.masked_fill(attention_mask, -np.inf)
        if attention_mask is not None:
            att = att.masked_fill(attention_mask == 0, float('-inf'))
        

        # get attention matrix
        att = torch.softmax(att, -1)
        att = self.attn_drop(att)

        # output
        out = torch.matmul(att, v).permute(0, 2, 1, 3).contiguous().view(b_s, nq, self.h * self.d_v)  # (b_s, nq, h*d_v)
        out = self.resid_drop(self.out_proj(out))  # (b_s, nq, d_model)

        return out


class myTransformerBlock(nn.Module):
    def __init__(self, d_model, d_k, d_v, h, block_exp, attn_pdrop, resid_pdrop, lora_rank=32, lora_alpha=2.0, lora_beta=0.0):
        super().__init__()
        self.ln_input = nn.LayerNorm(d_model)
        self.ln_output = nn.LayerNorm(d_model)

        self.sa = SelfAttention(d_model, d_k, d_v, h, attn_pdrop, resid_pdrop, lora_rank, lora_alpha, lora_beta)
        
        # 更新MLP以使用LoRA
        self.mlp = nn.Sequential(
            QALoRALayer(d_model, block_exp * d_model, lora_rank, d_model // h, s=lora_alpha, alpha=lora_alpha, beta=lora_beta),
            nn.GELU(),
            QALoRALayer(block_exp * d_model, d_model, lora_rank, d_model // h, s=lora_alpha, alpha=lora_alpha, beta=lora_beta),
            nn.Dropout(resid_pdrop),
        )

    def forward(self, x):

        x = x + self.sa(self.ln_input(x))
        x = x + self.mlp(self.ln_output(x))

        return x


class GPT_QALORA(nn.Module):
    """  the full GPT language model, with a context size of block_size """

    def __init__(self, d_model, h=8, block_exp=4, n_layer=8, vert_anchors=8, horz_anchors=8,
                 embd_pdrop=0.2, attn_pdrop=0.2, resid_pdrop=0.2, lora_rank=32, lora_alpha=2.0):
        
        super().__init__()

        self.n_embd = d_model

        self.vert_anchors = vert_anchors
        self.horz_anchors = horz_anchors

        d_k = d_model
        d_v = d_model

        # positional embedding parameter (learnable), rgb_fea + ir_fea
        self.pos_emb = nn.Parameter(torch.zeros(1, 2 * vert_anchors * horz_anchors, self.n_embd))

        # transformer
        self.trans_blocks = nn.Sequential(*[
            myTransformerBlock(d_model, d_model // h, d_model // h, h, block_exp, attn_pdrop, resid_pdrop, lora_rank, lora_alpha)
            for _ in range(n_layer)])

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

