import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter


class MultiGraphConvolution_Layer(nn.Module):

    def __init__(self, in_features, out_features, support_num):
        super(MultiGraphConvolution_Layer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.support_num = support_num

        self.weight = Parameter(torch.Tensor(support_num, in_features, out_features))
        self.layer_attention = Parameter(torch.Tensor(support_num))
        self.bias = Parameter(torch.Tensor(out_features))
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.normal_(self.weight, mean=0, std=1)
        torch.nn.init.normal_(self.layer_attention, mean=0, std=1)
        torch.nn.init.normal_(self.bias, mean=0, std=1)

    def forward(self, support_matmul_input):
        supports = support_matmul_input.matmul(self.weight)
        supports = supports.permute(1, 2, 0)
        # supports = F.relu(supports)
        output = torch.mul(self.layer_attention, supports).sum(dim=2) + self.bias

        return output


class MGC_Model(nn.Module):

    def __init__(self, feature_num, hidden_num, support_num, target_num, only_cell_type=False):
        """
        Multi-view graph convolutional model.
        Feature matrix is the cell-type matrix (cell_num * cell_type_num).
        Adjacency matrix is the preprocessed CE tensor (LR_pair_num * cell_num * cell_num).
        Output is the target gene expression matrix (cell_num * target_gene_num).

        Parameters
        ----------
        feature_num :
            The dim of the feature matrix.
        hidden_num :
            The dim of 'MultiGraphConvolution_Layer' output. Always use 1 or same as feature_num.
        support_num :
            The number of view in the multi-view CE network, same as LR_pair_num
        target_num :
            The dim of the model output, same as the target_gene_num. Always use 1 and generate one gene for one time.
        only_cell_type :
             If true, the model only use the Feature matrix training target, serving as a baseline model.

        """
        super(MGC_Model, self).__init__()
        self.feature_num = feature_num
        self.hidden_num = hidden_num  # hidden_num = 1 or feature_num
        self.support_num = support_num
        self.only_cell_type = only_cell_type

        if self.only_cell_type:
            self.linear_b = nn.Linear(feature_num, target_num)
        else:
            self.mgc = MultiGraphConvolution_Layer(in_features=feature_num, out_features=hidden_num,
                                                   support_num=support_num, )
            self.linear_b = nn.Linear(feature_num, target_num)
            self.linear_ce = nn.Linear(hidden_num, target_num)

    def forward(self, input_x, adj_matmul_input_x):

        if self.only_cell_type:
            x = self.linear_b(input_x)
        else:
            x = self.mgc(adj_matmul_input_x)
            x = F.relu(x)
            
            x_ce = self.linear_ce(x)
            x_b = self.linear_b(input_x)
            x = x_b + x_ce
            
        x = torch.sigmoid(x)
        return x
