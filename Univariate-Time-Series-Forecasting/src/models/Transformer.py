import numpy as np
import torch
import torch.nn as nn

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        # pe.requires_grad = False
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor):
        chunk = x.chunk(x.size(-1), dim=2)
        out = torch.Tensor([]).to(x.device)
        for i in range(len(chunk)):
            out = torch.cat((out, chunk[i] + self.pe[:chunk[i].size(0), ...]), dim=2)
        return out
def transformer_generate_tgt_mask(length, device):
    mask = torch.tril(torch.ones(length, length, device=device)) == 1
    mask = (
        mask.float()
            .masked_fill(mask == 0, float("-inf"))
            .masked_fill(mask == 1, float(0.0))
    )
    return mask

class TransformerModel(nn.Module):
    """标准的Transformer编码器-解码器结构"""
    def __init__(self, n_encoder_inputs, n_decoder_inputs,num_heads,Sequence_length,d_model, dropout, num_layer):
        """
        初始化
        :param n_encoder_inputs:    输入数据的特征维度
        :param n_decoder_inputs:    编码器输入的特征维度，其实等于编码器输出的特征维度
        :param d_model:             词嵌入特征维度
        :param dropout:             dropout
        :param num_layer:           Transformer块的个数
         Sequence_length:           transformer 输入数据 序列的长度
        """
        super(TransformerModel, self).__init__()

        self.input_pos_embedding = torch.nn.Embedding(5000, embedding_dim=d_model)
        encoder_layer = torch.nn.TransformerEncoderLayer(d_model=d_model, nhead=num_heads, dropout=dropout,
                                                         dim_feedforward=4 * d_model)
        self.encoder = torch.nn.TransformerEncoder(encoder_layer, num_layers=num_layer)
        self.input_projection = torch.nn.Linear(n_encoder_inputs, d_model)


        self.target_pos_embedding = torch.nn.Embedding(5000, embedding_dim=d_model)
        decoder_layer = torch.nn.TransformerDecoderLayer(d_model=d_model, nhead=num_heads, dropout=dropout,
                                                         dim_feedforward=4 * d_model)
        self.decoder = torch.nn.TransformerDecoder(decoder_layer, num_layers=num_layer)
        self.output_projection = torch.nn.Linear(n_decoder_inputs, d_model)

        self.linear = torch.nn.Linear(d_model, 1)
        self.ziji_add_linear=torch.nn.Linear(Sequence_length, 1)

    def encode_in(self, src):
        src_start = self.input_projection(src).permute(1, 0, 2)
        in_sequence_len, batch_size = src_start.size(0), src_start.size(1)
        pos_encoder = (torch.arange(0, in_sequence_len, device=src.device).unsqueeze(0).repeat(batch_size, 1))
        pos_encoder = self.input_pos_embedding(pos_encoder).permute(1, 0, 2)
        src = src_start + pos_encoder
        src = self.encoder(src) + src_start
        return src
    def decode_out(self, tgt, memory):
        tgt = tgt.long()
        tgt_start = self.target_pos_embedding(tgt).permute(1, 0, 2)
        out_sequence_len, batch_size = tgt_start.size(0), tgt_start.size(1)
        pos_decoder = (torch.arange(0, out_sequence_len, device=tgt.device).unsqueeze(0).repeat(batch_size, 1))
        pos_decoder = self.target_pos_embedding(pos_decoder).permute(1, 0, 2)
        tgt = tgt_start + pos_decoder
        tgt_mask = transformer_generate_tgt_mask(out_sequence_len, tgt.device)
        out = self.decoder(tgt=tgt, memory=memory, tgt_mask=tgt_mask) + tgt_start
        out = out.permute(1, 0, 2)  # [batch_size, seq_len, d_model]
        out = self.linear(out)
        return out

    def forward(self, src, target_in):
        src = self.encode_in(src)
        out = self.decode_out(tgt=target_in, memory=src)
        # print("out.shape:",out.shape)# torch.Size([batch, 3, 1]) # 原本代码中的输出
        # 上边的这个输入可以用于很多任务的输出 可以根据任务进行自由的变换
        # 下面是自己修改的
        # 使用全连接变成 [batch,1] 构成了基于transformer的回归单值预测
        out=out.squeeze(2)
        #out=self.ziji_add_linear(out)
        return out

