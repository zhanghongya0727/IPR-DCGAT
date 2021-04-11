import torch.nn as nn
import torch
import torch.nn.functional as F
import math


class GraphConvLayer(nn.Module):
    def __init__(self, input_dim, layers_num, dropout, device):
        super(GraphConvLayer, self).__init__()
        self.input_dim = input_dim
        self.layers_num = layers_num
        self.single_dim = self.input_dim // self.layers_num
        self.device = device
        self.gcn_drop = nn.Dropout(dropout)
        self.linear_output = nn.Linear(self.input_dim, self.input_dim).to(self.device)
        self.weight_list = nn.ModuleList()
        for i in range(self.layers_num):
            self.weight_list.append(nn.Linear((self.input_dim + self.single_dim * i), self.single_dim))
        self.weight_list = self.weight_list.to(self.device)

    def forward(self, gcn_inputs, adj):
        outputs = gcn_inputs
        cache_list = [outputs]
        output_list = []
        for l in range(self.layers_num):
            Ax = adj.bmm(outputs)
            AxW = self.weight_list[l](Ax)  # 变成single_dim
            AxW = F.relu(AxW)
            cache_list.append(AxW)
            outputs = torch.cat(cache_list, dim=2)  # 前n层的隐藏层输出+输入作为下一层的输入
            output_list.append(self.gcn_drop(AxW))  # 连接所有隐藏层的输出
        gcn_outputs = torch.cat(output_list, dim=2)
        gcn_outputs = gcn_outputs + gcn_inputs
        out = self.linear_output(gcn_outputs)
        return out


class MultiGraphConvLayer(nn.Module):
    def __init__(self, input_dim, layers_num, dropout, heads, device):
        super(MultiGraphConvLayer, self).__init__()
        self.input_dim = input_dim
        self.layers_num = layers_num
        self.single_dim = self.input_dim // self.layers_num
        self.heads = heads
        self.device = device
        self.gcn_drop = nn.Dropout(dropout)
        self.linear_output = nn.Linear(self.input_dim * self.heads, self.input_dim).to(self.device)
        self.weight_list = nn.ModuleList()
        for j in range(heads):
            for i in range(self.layers_num):
                self.weight_list.append(nn.Linear((self.input_dim + self.single_dim * i), self.single_dim))
        self.weight_list = self.weight_list.to(self.device)

    def forward(self, gcn_inputs, adj):
        multi_head_list = []
        for j in range(self.heads):
            outputs = gcn_inputs
            cache_list = [outputs]
            output_list = []
            for l in range(self.layers_num):
                index = j * self.layers_num + l
                Ax = adj.bmm(outputs)
                AxW = self.weight_list[index](Ax)  # 变成single_dim
                AxW = F.relu(AxW)
                cache_list.append(AxW)
                outputs = torch.cat(cache_list, dim=2)  # 前n层的隐藏层输出+输入作为下一层的输入
                output_list.append(self.gcn_drop(AxW))  # 连接所有隐藏层的输出
            gcn_outputs = torch.cat(output_list, dim=2)
            gcn_outputs = gcn_outputs + gcn_inputs
            multi_head_list.append(gcn_outputs)
        out = self.linear_output(torch.cat(multi_head_list, dim=2))
        return out


class GraphAttentionLayer(nn.Module):
    def __init__(self, input_dim, layers_num, dropout, alpha, device):
        super(GraphAttentionLayer, self).__init__()
        self.input_dim = input_dim
        self.layers_num = layers_num
        self.single_dim = self.input_dim // self.layers_num
        self.alpha = alpha
        self.device = device
        self.dropout = nn.Dropout(dropout)
        self.linear_output = nn.Linear(self.input_dim, self.input_dim).to(self.device)
        self.weight_list = nn.ModuleList()
        for i in range(self.layers_num):
            self.weight_list.append(nn.Linear((self.input_dim + self.single_dim * i), self.single_dim))
        self.weight_list = self.weight_list.to(self.device)
        self.a = nn.Parameter(torch.zeros(size=(2 * self.single_dim, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, gcn_inputs, adj):
        outputs = gcn_inputs
        cache_list = [outputs]
        output_list = []
        for l in range(self.layers_num):
            xW = self.weight_list[l](outputs)
            N = xW.size()[1]
            r_idx, c_idx = torch.meshgrid(torch.arange(N).to(self.device), torch.arange(N).to(self.device))
            a_input = torch.cat((xW[:, r_idx], xW[:, c_idx]), dim=3)  # [batch_size, node_num, node_num, dim*2]
            e = self.leakyrelu(
                torch.matmul(a_input, self.a).squeeze(3))  # LeakyReLU(a_input,a) [batch_size, node_num, node_num]
            zero_vec = -9e15 * torch.ones_like(e)
            attention = torch.where(adj > 0, e, zero_vec)  # shape [batch_size, node_num, node_num]
            attention = F.softmax(attention, dim=2)
            AxW = attention.bmm(xW)  # shape [batch_size, node_num, dim]  40 -> 40 -> 40
            AxW = F.relu(AxW)
            cache_list.append(AxW)
            outputs = torch.cat(cache_list, dim=2)  # 前n层的隐藏层输出+输入作为下一层的输入
            output_list.append(self.dropout(AxW))  # 连接所有隐藏层的输出
        gcn_outputs = torch.cat(output_list, dim=2)
        gcn_outputs = gcn_outputs + gcn_inputs
        out = self.linear_output(gcn_outputs)
        return out


class MultiGraphAttentionLayer(nn.Module):
    def __init__(self, input_dim, layers_num, dropout, heads, alpha, device):
        super(MultiGraphAttentionLayer, self).__init__()
        self.input_dim = input_dim
        self.layers_num = layers_num
        self.single_dim = self.input_dim // self.layers_num
        self.heads = heads
        self.alpha = alpha
        self.device = device
        self.dropout = nn.Dropout(dropout)
        self.linear_output = nn.Linear(self.input_dim * self.heads, self.input_dim).to(self.device)
        self.weight_list = nn.ModuleList()
        for j in range(heads):
            for i in range(self.layers_num):
                self.weight_list.append(nn.Linear((self.input_dim + self.single_dim * i), self.single_dim))
        self.weight_list = self.weight_list.to(self.device)
        self.a = nn.Parameter(torch.zeros(size=(2 * self.single_dim, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, gcn_inputs, adj):
        multi_head_list = []
        for j in range(self.heads):
            outputs = gcn_inputs
            cache_list = [outputs]
            output_list = []
            for l in range(self.layers_num):
                index = j * self.layers_num + l
                xW = self.weight_list[index](outputs)
                N = xW.size()[1]
                r_idx, c_idx = torch.meshgrid(torch.arange(N).to(self.device), torch.arange(N).to(self.device))
                a_input = torch.cat((xW[:, r_idx], xW[:, c_idx]), dim=3)  # [batch_size, node_num, node_num, dim*2]
                e = self.leakyrelu(
                    torch.matmul(a_input, self.a).squeeze(3))  # LeakyReLU(a_input,a) [batch_size, node_num, node_num]
                zero_vec = -9e15 * torch.ones_like(e)
                attention = torch.where(adj > 0, e, zero_vec)  # shape [batch_size, node_num, node_num]
                attention = F.softmax(attention, dim=2)
                AxW = attention.bmm(xW)  # shape [batch_size, node_num, dim]  40 -> 40 -> 40
                AxW = F.relu(AxW)
                cache_list.append(AxW)
                outputs = torch.cat(cache_list, dim=2)  # 前n层的隐藏层输出+输入作为下一层的输入
                output_list.append(self.dropout(AxW))  # 连接所有隐藏层的输出
            gcn_outputs = torch.cat(output_list, dim=2)
            gcn_outputs = gcn_outputs + gcn_inputs
            multi_head_list.append(gcn_outputs)
        out = self.linear_output(torch.cat(multi_head_list, dim=2))
        return out


class GraphAttentionLayer_old(nn.Module):

    def __init__(self, in_features, out_features, dropout, alpha, concat=True, device=-1):
        super(GraphAttentionLayer_old, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat
        self.device = device
        self.dropout = dropout
        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.zeros(size=(2 * out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        self.leakyrelu = nn.LeakyReLU(self.alpha)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input, adj):
        h = torch.matmul(input, self.W)  # adj shape [3,72,72] input*w shape[3,72,210]
        N = h.size()[1]
        r_idx, c_idx = torch.meshgrid(torch.arange(N).to(self.device), torch.arange(N).to(self.device))
        a_input = torch.cat((h[:, r_idx], h[:, c_idx]), dim=3)  # shape[3,72,72,420]
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(3))  # LeakyReLU(a_input,a) shape[3,72,72]
        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)  # shape[3,72,72]
        attention = F.softmax(attention, dim=2)
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, h)  # shape[3,72,110]
        return h_prime


class GraphConvolution_old(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution_old, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))  # shape[110,110]
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.matmul(input, self.weight)
        output = torch.matmul(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output
