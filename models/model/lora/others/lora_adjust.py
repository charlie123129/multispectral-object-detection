import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.nn import init, Sequential
import math


class LoRALayer(nn.Module):
    def __init__(self, in_features, out_features, rank, alpha=1.0):
        super(LoRALayer, self).__init__()
        self.rank = rank  # LoRA层的秩，控制低秩矩阵的大小
        self.alpha = alpha  # 调整LoRA效应的缩放因子

        # Original weight and bias
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))  # 原始权重矩阵
        self.bias = nn.Parameter(torch.Tensor(out_features))  # 原始偏置向量

        # Low-rank matrices
        self.delta_W = nn.Parameter(torch.Tensor(rank, in_features))  # 低秩矩阵W
        self.delta_B = nn.Parameter(torch.Tensor(out_features, rank))  # 低秩矩阵B

        self.reset_parameters()  # 初始化参数


    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5)) # 使用Kaiming初始化权重
        nn.init.zeros_(self.bias) # 初始化偏置为0
        # 调整delta_W和delta_B的初始化方法，根据实验结果选择最佳标准差
        nn.init.normal_(self.delta_W, std=0.01)  # 假设调整为0.01
        nn.init.normal_(self.delta_B, std=0.01)  # 假设调整为0.01


    def forward(self, x):
        W = self.weight + self.alpha * (self.delta_B @ self.delta_W)  # 计算调整后的权重矩阵
        return F.linear(x, W, self.bias)  # 应用线性变换


class SelfAttention(nn.Module):
    """
     Multi-head masked self-attention layer
    """
    def __init__(self, d_model, d_k, d_v, h, attn_pdrop=0.2, resid_pdrop=0.2, lora_rank=32, lora_alpha=1.0):
        super(SelfAttention, self).__init__()
        assert d_k % h == 0  # 确保每个头的维度能够整除
        self.d_model = d_model  # 模型的维度
        self.d_k = d_model // h  # 每个头的键、查询的维度
        self.d_v = d_model // h  # 每个头的值的维度
        self.h = h  # 头的数量

        # key, query, value projections for all heads
        self.que_proj = LoRALayer(d_model, h * self.d_k, lora_rank, lora_alpha)  # 查询投影
        self.key_proj = LoRALayer(d_model, h * self.d_k, lora_rank, lora_alpha)  # 键投影
        self.val_proj = LoRALayer(d_model, h * self.d_v, lora_rank, lora_alpha)  # 值投影
        self.out_proj = LoRALayer(h * self.d_v, d_model, lora_rank, lora_alpha)  # 输出投影

        # regularization
        self.attn_drop = nn.Dropout(attn_pdrop)  # 注意力机制的dropout
        self.resid_drop = nn.Dropout(resid_pdrop)  # 残差连接的dropout

        self.init_weights()  # 初始化权重


    def init_weights(self):
        # 初始化权重的函数，对不同类型的层使用不同的初始化策略
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
        # 计算自注意力的函数
        b_s, nq = x.shape[:2]  # 批大小和查询的长度
        nk = x.shape[1]  # 键的长度
        q = self.que_proj(x).view(b_s, nq, self.h, self.d_k).permute(0, 2, 1, 3)  # 查询矩阵
        k = self.key_proj(x).view(b_s, nk, self.h, self.d_k).permute(0, 2, 3, 1)  # 键矩阵
        v = self.val_proj(x).view(b_s, nk, self.h, self.d_v).permute(0, 2, 1, 3)  # 值矩阵

        # 计算注意力得分
        att = torch.matmul(q, k) / np.sqrt(self.d_k)  # 注意力得分

        # 应用注意力掩码（如果有）
        if attention_mask is not None:
            att = att.masked_fill(attention_mask == 0, float('-inf'))
        
        # 应用softmax并进行dropout处理
        att = torch.softmax(att, -1)
        att = self.attn_drop(att)

        # 计算输出
        out = torch.matmul(att, v).permute(0, 2, 1, 3).contiguous().view(b_s, nq, self.h * self.d_v)  # 最终注意力输出
        out = self.resid_drop(self.out_proj(out))  # 应用残差dropout和输出投影

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
        self.ln_input = nn.LayerNorm(d_model)  # 输入的层归一化
        self.ln_output = nn.LayerNorm(d_model)  # 输出的层归一化
        self.sa = SelfAttention(d_model, d_k, d_v, h, attn_pdrop, resid_pdrop)  # 自注意力模块
        # MLP层，使用LoRA层进行参数调整，增加模型的表达能力
        self.mlp = nn.Sequential(
            LoRALayer(d_model, block_exp * d_model, lora_rank, lora_alpha),  # 扩展的线性变换
            nn.GELU(),  # 激活函数
            LoRALayer(block_exp * d_model, d_model, lora_rank, lora_alpha),  # 再次缩放到原始维度
            nn.Dropout(resid_pdrop),  # 避免过拟合的dropout层
        )

    def forward(self, x):
        """
        前向传播函数定义了数据通过Transformer块的流程。
        """
        x = x + self.sa(self.ln_input(x))  # 应用自注意力机制前的层归一化和残差连接
        x = x + self.mlp(self.ln_output(x))  # 应用MLP前的层归一化和残差连接

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
