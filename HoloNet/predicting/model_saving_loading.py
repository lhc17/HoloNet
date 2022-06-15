import os
from typing import List, Optional, Tuple, Union

import torch

from .MGC_model import MGC_Model


def save_model_list(trained_MGC_model_list: List[MGC_Model],
                    project_name: str,
                    target_gene_name_list: List[str],
                    model_save_folder: str = '_tmp_save_model',
                    ):
    """

    Save the trained model list in model_save_folder/project_name/target_gene_name_list

    Parameters
    ----------
    trained_MGC_model_list :
        A list of trained MGC model for generating the expression of one target gene.
    project_name :
        The name of project, such as 'BRCA_10x_generating_all_target_gene'.
    target_gene_name_list :
        The target gene name list.
    model_save_folder :
        The father folder.

    """

    for gene_i, target_gene_name in enumerate(target_gene_name_list):
        model_save_dir = os.path.join(model_save_folder, project_name, target_gene_name)
        if not os.path.exists(model_save_dir):
            os.makedirs(model_save_dir)
        
        if len(target_gene_name_list) != 1:
            trained_MGC_model_list_i = trained_MGC_model_list[gene_i]
        else:
            trained_MGC_model_list_i = trained_MGC_model_list
            
        for model_i, model in enumerate(trained_MGC_model_list_i):
            model_save_path = os.path.join(model_save_dir, str(model_i) + 'th_trained_MGC_model.pt')
            torch.save(model.state_dict(), model_save_path)


def load_model_list(X: torch.Tensor,
                    adj: torch.Tensor,
                    project_name: str,
                    used_target_gene_name_list: Optional[str] = None,
                    only_cell_type: bool = False,
                    model_save_folder: str = '_tmp_save_model',
                    ) -> Tuple[Union[List[MGC_Model], List[List[MGC_Model]]], List[str]]:
    """

    Load trained model list in model_save_folder/project_name/

    Parameters
    ----------
    X :
        The feature matrix used as the input of the loading trained models.
    adj :
        The adjacency matrix used as the input of the loading trained models.
    project_name :
        The name of project, such as 'BRCA_10x_generating_all_target_gene'.
    used_target_gene_name_list :
        If not None, can only load part of models for some given genes.
    only_cell_type :
        If true, the model only use the Feature matrix training target, serving as a baseline model.
    model_save_folder :
        The father folder.

    Returns
    -------
    trained_MGC_model_list_all_target :
        A list of trained MGC model for generating the expression of one target gene.
        Or a list of the trained MGC model lists for multiple target genes, if with multiple targets.
    all_target_gene_list :
        A list of target gene names.


    """

    trained_MGC_model_list_all_target = []

    if used_target_gene_name_list is None:
        all_target_gene_list = os.listdir(os.path.join(model_save_folder, project_name))
    else:
        all_target_gene_list = [used_target_gene_name_list]

    for used_target_gene_name_list in all_target_gene_list:
        trained_MGC_model_list = []
        model_save_dir = os.path.join(model_save_folder, project_name, used_target_gene_name_list)
        if not os.path.exists(model_save_dir):
            print('Not found the target folder')

        files = os.listdir(model_save_dir)
        for file in files:
            model = MGC_Model(feature_num=X.shape[1],
                              hidden_num=X.shape[1],
                              support_num=adj.shape[0],
                              target_num=1,
                              only_cell_type=only_cell_type)
            loaded_paras = torch.load(os.path.join(model_save_dir, file))
            model.load_state_dict(loaded_paras)
            trained_MGC_model_list.append(model)
        trained_MGC_model_list_all_target.append(trained_MGC_model_list)

    if len(trained_MGC_model_list_all_target) == 1:
        trained_MGC_model_list_all_target = trained_MGC_model_list_all_target[0]

    return trained_MGC_model_list_all_target, all_target_gene_list
