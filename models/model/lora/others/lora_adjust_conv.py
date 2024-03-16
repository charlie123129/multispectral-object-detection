import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.nn import init, Sequential
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


class ConvLoRALayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, rank, alpha=1.0, stride=1, padding=0, dilation=1):
        super(ConvLoRALayer, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation)
        self.lora = LoRALayer(out_channels, out_channels, rank, alpha)
        self.rank = rank
        self.alpha = alpha

    def forward(self, x):
        # 先进行卷积操作
        conv_out = self.conv(x)
        
        # 将卷积输出的每个特征图看作是一个独立的样本，应用LoRA调整
        # 假设conv_out的形状为[batch_size, out_channels, H, W]
        # 需要将其变换为[batch_size*H*W, out_channels]来应用LoRA层
        b, c, h, w = conv_out.size()
        conv_out_flat = conv_out.permute(0, 2, 3, 1).reshape(-1, c)
        
        # 应用LoRA调整
        lora_out_flat = self.lora(conv_out_flat)
        
        # 将输出恢复到原始的四维形状
        lora_out = lora_out_flat.reshape(b, h, w, c).permute(0, 3, 1, 2)
        
        return lora_out


class SelfAttention(nn.Module):
    """
     Multi-head masked self-attention layer
    """
    def __init__(self, d_model, d_k, d_v, h, attn_pdrop=0.2, resid_pdrop=0.2, lora_rank=32, lora_alpha=1.0):
        '''
        :param d_model: Output dimensionality of the model
        :param d_k: Dimensionality of queries and keys
        :param d_v: Dimensionality of values
        :param h: Number of heads
        '''
        super(SelfAttention, self).__init__()
        assert d_k % h == 0
        self.d_model = d_model
        self.d_k = d_model // h
        self.d_v = d_model // h
        self.h = h

        # key, query, value projections for all heads
        self.que_proj = LoRALayer(d_model, h * self.d_k, lora_rank, lora_alpha)  # query projection
        self.key_proj = LoRALayer(d_model, h * self.d_k, lora_rank, lora_alpha)  # key projection
        self.val_proj = LoRALayer(d_model, h * self.d_v, lora_rank, lora_alpha)  # value projection
        self.out_proj = LoRALayer(h * self.d_v, d_model, lora_rank, lora_alpha)  # output projection

        # regularization
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
    """ Transformer block """
    def __init__(self, d_model, d_k, d_v, h, block_exp, attn_pdrop, resid_pdrop, lora_rank=32, lora_alpha=2.0):
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
        self.sa = SelfAttention(d_model, d_k, d_v, h, attn_pdrop, resid_pdrop)
        # 更新MLP以使用LoRA
        self.mlp = nn.Sequential(
            LoRALayer(d_model, block_exp * d_model, lora_rank, lora_alpha),
            nn.GELU(),
            LoRALayer(block_exp * d_model, d_model, lora_rank, lora_alpha),
            nn.Dropout(resid_pdrop),
        )

    def forward(self, x):

        x = x + self.sa(self.ln_input(x))
        x = x + self.mlp(self.ln_output(x))

        return x


class GPT_LORA_ADJUST_CONV(nn.Module):
    def __init__(self, in_channels, d_model, h=8, block_exp=4, n_layer=8, vert_anchors=8, horz_anchors=8,
                 embd_pdrop=0.2, attn_pdrop=0.2, resid_pdrop=0.2, lora_rank=32, lora_alpha=2.0, kernel_size=3, stride=1, padding=1):
        super().__init__()

        self.conv_lora = ConvLoRALayer(in_channels, d_model, kernel_size, lora_rank, lora_alpha, stride, padding)  # 新增的ConvLoRALayer
        
        self.n_embd = d_model
        self.vert_anchors = vert_anchors
        self.horz_anchors = horz_anchors

        self.pos_emb = nn.Parameter(torch.zeros(1, 2 * vert_anchors * horz_anchors, self.n_embd))

        self.current_alpha = lora_alpha  # 先直接存储alpha值，不调用adjust_alpha
        
        # Transformer块序列
        self.trans_blocks = nn.Sequential(*[
            myTransformerBlock(d_model, d_model // h, d_model // h, h, block_exp, attn_pdrop, resid_pdrop, lora_rank, self.current_alpha)
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
                if isinstance(module, LoRALayer):
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
        rgb_fea, ir_fea = x[0], x[1]
        bs, c, h, w = rgb_fea.shape
        
        # 使用ConvLoRALayer处理RGB和IR特征
        rgb_fea = self.conv_lora(rgb_fea)
        ir_fea = self.conv_lora(ir_fea)
        
        # 接下来的步骤与原始GPT_LORA类似
        #rgb_fea = self.avgpool(rgb_fea)
        #ir_fea = self.avgpool(ir_fea)
        
        rgb_fea_flat = rgb_fea.view(bs, self.n_embd, -1)
        ir_fea_flat = ir_fea.view(bs, self.n_embd, -1)
        token_embeddings = torch.cat([rgb_fea_flat, ir_fea_flat], dim=2)
        token_embeddings = token_embeddings.permute(0, 2, 1).contiguous()

        x = self.drop(self.pos_emb + token_embeddings)
        x = self.trans_blocks(x)

        x = self.ln_f(x)
        x = x.view(bs, 2, self.vert_anchors, self.horz_anchors, self.n_embd)
        x = x.permute(0, 1, 4, 2, 3)

        rgb_fea_out = x[:, 0, :, :, :].contiguous().view(bs, self.n_embd, self.vert_anchors, self.horz_anchors)
        ir_fea_out = x[:, 1, :, :, :].contiguous().view(bs, self.n_embd, self.vert_anchors, self.horz_anchors)

        rgb_fea_out = F.interpolate(rgb_fea_out, size=([h, w]), mode='bilinear')
        ir_fea_out = F.interpolate(ir_fea_out, size=([h, w]), mode='bilinear')

        return rgb_fea_out, ir_fea_out
