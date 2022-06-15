import torch
from sklearn.model_selection import ParameterGrid

class Hyperparameters:
    def __init__(self, model_type, device=None, patience=None,
                 du_list=None, clf_lr=None, clf_reg=None, re_lr=None, re_reg=None):

        # Initialization
        self.model_type = model_type
        self.param_list = None
        self.all_hyper_parameters_list = []

        self.du_list = du_list

        # Load Hyperparameters
        if model_type == 'SVM':
            self.SVM()
        else:
            self.MMMF(device, patience, du_list, clf_lr, clf_reg, re_lr, re_reg)

    # SVM Hyperparameters
    def SVM(self):
        param_grid = {'gamma': ['scale'], 'random_state': [3], 'probability': [True],
                      'C': [1, 10, 100], 'kernel': ['poly', 'linear', 'rbf']}
        self.param_list = list(ParameterGrid(param_grid))

    # MMMF Based Hyperparameters
    def MMMF(self, device, patience, du_list, clf_lr, clf_reg, re_lr, re_reg):
        param_list = []

        for i, d1 in enumerate(du_list):
            du_list2 = du_list[i + 1:]
            for j, d2 in enumerate(du_list2):
                du_list3 = du_list2[j + 1:]
                # 50000
                for d3 in du_list3:
                    hyper_dict = {'patience': patience, 'epoch': 3,
                                  'device': torch.device("cuda:" + str(device) if torch.cuda.is_available() else "cpu"),
                                  'du1': d1, 'du2': d2, 'du3': d3,
                                  'clf_lr': clf_lr, 'clf_reg': clf_reg, 're_lr': re_lr, 're_reg': re_reg}

                    param_list.append(hyper_dict)

        self.param_list = param_list

    def all_hyper_parameters(self):
        all_hyper_parameters_list = []

        if self.model_type == 'SVM':
            for C in [1, 10, 100]:
                for kernel in ['poly', 'linear', 'rbf']:
                    all_hyper_parameters_list.append(str(C) + '_' + kernel)

        else:
            # Hyperparameter - Du1, Du2 Dimension
            for i, d1 in enumerate(self.du_list):
                du_list2 = self.du_list[i + 1:]
                for j, d2 in enumerate(du_list2):
                    du_list3 = du_list2[j + 1:]
                    for d3 in du_list3:
                        all_hyper_parameters_list.append(str(d1) + '_' + str(d2) + '_' + str(d3))

        return all_hyper_parameters_list

    def one_hyper_parameters(self, hyper_parameter_dict):
        if self.model_type == 'SVM':
            return str(hyper_parameter_dict['C']) + '_' + hyper_parameter_dict['kernel']

        else:
            return str(hyper_parameter_dict['du1']) + '_' + str(hyper_parameter_dict['du2']) + '_' + str(hyper_parameter_dict['du3'])
