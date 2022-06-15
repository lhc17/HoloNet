from copy import deepcopy
from typing import Optional, List

import numpy as np
import torch
from torch import optim
from torch.nn import functional as F
from torch.optim import lr_scheduler
from tqdm import tqdm
import pynvml
import random
import os

from .MGC_model import MGC_Model
from .input_preprocessing import train_test_mask



def mgc_repeat_training(X: torch.Tensor,
                        adj: torch.Tensor,
                        target: torch.Tensor,
                        repeat_num: int = 50,
                        train_set_ratio: float = 0.85,
                        val_set_ratio: float = 0.15,
                        hidden_num: Optional[int] = None,
                        max_epoch: int = 300,
                        lr: float = 0.1,
                        weight_decay: float = 5e-4,
                        step_size: int = 10,
                        gamma: float = 0.9,
                        display_loss: bool = False,
                        only_cell_type: bool = False,
                        hide_repeat_tqdm: bool = False,
                        device: str = 'cpu',
                        ) -> List[MGC_Model]:
    """

    Using cell-type tensor and normalized adjancency matrix as the inputs, repeated training GNN to generate the target
    gene expression.

    Parameters
    ----------
    X :
        A tensor (cell_num * cell_type_num) with cell-type information. derived from 'get_continuous_cell_type_tensor'
        or 'get_one_hot_cell_type_tensor' function.
    adj :
        A normalized adjancency matrix derived from 'adj_normalize' function.
    target :
        The scaled expression tensor of one target gene (cell_num * 1), derived from 'get_one_case_expr' function.
    repeat_num :
        The number of repeated training, defaultly as 50.
    train_set_ratio :
        A value from 0-1. The ratio of cells using as the training set.
    val_set_ratio :
        A value from 0-1. The ratio of cells using as the validation set.
    hidden_num :
        The dim of 'MultiGraphConvolution_Layer' output. Always use 1 or same as feature_num.
    max_epoch :
        The maximum epoch of training/
    lr :
        The learning rate.
    weight_decay :
        The weight decay (L2 penalty)
    step_size :
        Period of learning rate decay.
    gamma :
        Multiplicative factor of learning rate decay.
    display_loss :
        If true, display the loss during training.
    only_cell_type :
        If true, the model only use the Feature matrix training target, serving as a baseline model.
    hide_repeat_tqdm :
        If true, hide the tqdm for repeated training.
    use_gpu :
        If true, model will be trained in GPU when GPU is available.
    device :
        Give a device to use

    Returns
    -------
    A list of trained MGC model for generating the expression of one target gene.

    """
    
    seed_torch()
    if hidden_num is None:
        hidden_num = X.shape[1]
    if len(target.shape) == 1:
        target = target.unsqueeze(1)
        
    device = get_device(device)

    train_mask, test_mask, val_mask = train_test_mask(X.shape[0], train_set_ratio=train_set_ratio,
                                                      val_set_ratio=val_set_ratio)

    trained_MGC_model_list = []
    for i, _ in enumerate(tqdm(range(repeat_num), disable=hide_repeat_tqdm)):
        trained_MGC_model = mgc_training(X, adj, target, display=display_loss,
                                         train_mask=train_mask, test_mask=test_mask,
                                         hidden_num=hidden_num,
                                         val_mask=val_mask, only_cell_type=only_cell_type,
                                         max_epoch=max_epoch, lr=lr, weight_decay=weight_decay,
                                         step_size=step_size, gamma=gamma, device=device, seed=i)
        trained_MGC_model_list.append(deepcopy(trained_MGC_model[0]))

    return trained_MGC_model_list


def mgc_training(X, adj, target_gene_expr, train_mask, test_mask, val_mask, seed=0,
                 only_cell_type=False, hidden_num=1, max_epoch=100, lr=0.1,
                 weight_decay=5e-4, step_size=10, gamma=0.9, display=True, device='cpu'):

    seed_torch(seed)
    cell_type_num = X.shape[1]
    lr_pair_num = adj.shape[0]
    target_num = target_gene_expr.shape[1]
    target_gene_expr = target_gene_expr.to(device)

    MGC_model = MGC_Model(feature_num=cell_type_num,
                          hidden_num=hidden_num,
                          support_num=lr_pair_num,
                          target_num=target_num,
                          only_cell_type=only_cell_type).to(device)

    optimizer = optim.Adam(MGC_model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

    loss_list = []
    test_MSE_list = []
    val_MSE_list = []
    MGC_model_list = []
    
    if only_cell_type:
        adj_matmul_X = X.to(device)
    else:
        adj_matmul_X = adj.matmul(X).to(device)
    X = X.to(device)
    for epoch in range(max_epoch):
        z = MGC_model(X, adj_matmul_X)
        loss = F.mse_loss(z[train_mask, ], target_gene_expr[train_mask, ], reduction='mean').to(device)
        val_MSE = F.mse_loss(z[val_mask, ], target_gene_expr[val_mask, ], reduction='mean')
        test_MSE = F.mse_loss(z[test_mask, ], target_gene_expr[test_mask, ], reduction='mean')

        loss_list.append(loss.detach())
        test_MSE_list.append(test_MSE.detach().cpu())
        val_MSE_list.append(val_MSE.detach().cpu())
        MGC_model_list.append(deepcopy(MGC_model))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        if display:
            if epoch % 30 == 0:
                print('Epoch:', epoch)
                print('MSE on val dataset:', val_MSE)

    MGC_model = MGC_model_list[np.argmin(np.array(val_MSE_list))].cpu()
    return MGC_model, torch.stack(loss_list), torch.stack(test_MSE_list), torch.stack(val_MSE_list)


def get_mgc_result(trained_MGC_model_list: List[MGC_Model],
                   X: torch.Tensor,
                   adj: torch.Tensor,
                   hide_repeat_tqdm: bool = False,
                   device: str = 'cpu',
                   ) -> torch.Tensor:
    """

    Run the trained MGC model and get the generated expression profile of the target gene.

    Parameters
    ----------
    trained_MGC_model_list :
        A list of trained MGC model for generating the expression of one target gene.
    X :
        The feature matrix used as the input of the trained_MGC_model_list.
    adj :
        The adjacency matrix used as the input of the trained_MGC_model_list.
    hide_repeat_tqdm :
        If true, hide the tqdm for getting the result of repeated training.

    Returns
    -------
    The generated expression profile of the target gene (cell_num * 1).

    """
    
    device = get_device(device)
    
    repeat_num = len(trained_MGC_model_list)
    adj_matmul_X = adj.matmul(X).to(device)
    X = X.to(device)
    result = torch.hstack([trained_MGC_model_list[i].to(device)(X, adj_matmul_X).cpu()
                           for i in tqdm(range(repeat_num), disable=hide_repeat_tqdm)])
    result = result.mean(1).detach().unsqueeze(1)
    return result


def mgc_training_with_single_view(X: torch.Tensor,
                                  adj: torch.Tensor,
                                  target: torch.Tensor,
                                  device: str = 'cpu',
                                  repeat_num: int = 5,
                                  **kwargs,
                                  ) -> List[float]:
    """

    Parameters
    ----------
    X :
    adj :
    target :
    kwargs :

    Returns
    -------

    """
    
    device = get_device(device)
    coef_list = []
    for i in tqdm(range(adj.shape[0])):
        trained_MGC_model_list_tmp = mgc_repeat_training(X, adj[i].unsqueeze(0), target,
                                                         hide_repeat_tqdm=True, device=device, 
                                                         repeat_num=repeat_num, **kwargs)
        tmp = get_mgc_result(trained_MGC_model_list_tmp, X, adj[i].unsqueeze(0),
                             hide_repeat_tqdm=True)
        coef_list.append(np.corrcoef(tmp.squeeze(1).T, target.T)[0, 1])

    return coef_list


def get_device(device):
    if device != 'cpu' and torch.cuda.is_available():
        if ':' not in device:
            pynvml.nvmlInit()
            cuda_mem_free = []
            for i in range(torch.cuda.device_count()):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
                cuda_mem_free.append(meminfo.free)
            device = 'cuda:' + str(np.argmax(np.array(cuda_mem_free)))
    else:
        device = 'cpu'
        
    return device


def seed_torch(seed=100):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed) # 为了禁止hash随机化，使得实验可复现
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

