import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

class LoRALayer(nn.Module):
    def __init__(self, input_dim, output_dim, rank, h=1):
        super(LoRALayer, self).__init__()
        self.rank = rank
        self.h = h
        # Adjusting input_dim and output_dim based on h
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


class GPT_LORA_ADJUST(nn.Module):
    """  the full GPT language model, with a context size of block_size """

    def __init__(self, d_model, h=8, block_exp=4, n_layer=8, vert_anchors=8, horz_anchors=8,
                 embd_pdrop=0.2, attn_pdrop=0.2, resid_pdrop=0.2, lora_rank=32, lora_alpha=2.0):
        """
        构造函数初始化整个GPT模型及其组件。
        """
        super().__init__()

        self.n_embd = d_model  # 模型嵌入的维度

        self.vert_anchors = vert_anchors  # 垂直锚点数，用于确定位置嵌入的形状
        self.horz_anchors = horz_anchors  # 水平锚点数

        # 位置嵌入参数，可学习，对应于RGB和IR特征的维度
        self.pos_emb = nn.Parameter(torch.zeros(1, 2 * vert_anchors * horz_anchors, self.n_embd))

        self.current_alpha = lora_alpha  # 先直接存储alpha值，不调用adjust_alpha

        # Transformer块序列
        self.trans_blocks = nn.Sequential(*[
            myTransformerBlock(d_model, d_model // h, d_model // h, h, block_exp, attn_pdrop, resid_pdrop, lora_rank, self.current_alpha)
            for _ in range(n_layer)])
        
        self.adjust_alpha(lora_alpha)  # 在所有属性初始化之后调用adjust_alpha

        self.ln_f = nn.LayerNorm(self.n_embd)  # 输出的层归一化

        self.drop = nn.Dropout(embd_pdrop)  # 嵌入层的dropout

        self.avgpool = nn.AdaptiveAvgPool2d((self.vert_anchors, self.horz_anchors))  # 平均池化，降维

        self.apply(self._init_weights)  # 初始化模型权重
    
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
        """
        权重初始化函数。
        """
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, x):
        """
        前向传播函数定义了数据通过整个GPT模型的流程。
        """
        rgb_fea = x[0]  # RGB特征
        ir_fea = x[1]   # IR特征
        bs, c, h, w = rgb_fea.shape  # 获取特征形状

        rgb_fea = self.avgpool(rgb_fea)  # 对RGB特征进行平均池化
        ir_fea = self.avgpool(ir_fea)  # 对IR特征进行平均池化

        rgb_fea_flat = rgb_fea.view(bs, c, -1)  # 扁平化RGB特征
        ir_fea_flat = ir_fea.view(bs, c, -1)  # 扁平化IR特征
        token_embeddings = torch.cat([rgb_fea_flat, ir_fea_flat], dim=2)  # 合并特征
        token_embeddings = token_embeddings.permute(0, 2, 1).contiguous()  # 调整形状以适配Transformer输入

        x = self.drop(self.pos_emb + token_embeddings)  # 应用位置嵌入和dropout
        x = self.trans_blocks(x)  # 通过Transformer块序列

        x = self.ln_f(x)  # 应用输出的层归一化
        x = x.view(bs, 2, self.vert_anchors, self.horz_anchors, self.n_embd)  # 调整输出形状
        x = x.permute(0, 1, 4, 2, 3)  # 调整形状以分离RGB和IR输出

        rgb_fea_out = x[:, 0, :, :, :].contiguous().view(bs, self.n_embd, self.vert_anchors, self.horz_anchors)  # 获取RGB输出特征
        ir_fea_out = x[:, 1, :, :, :].contiguous().view(bs, self.n_embd, self.vert_anchors, self.horz_anchors)  # 获取IR输出特征

        rgb_fea_out = F.interpolate(rgb_fea_out, size=([h, w]), mode='bilinear')  # 上采样RGB输出特征
        ir_fea_out = F.interpolate(ir_fea_out, size=([h, w]), mode='bilinear')  # 上采样IR输出特征

        return rgb_fea_out, ir_fea_out
