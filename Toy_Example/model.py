from util import *
import copy
import torch
import numpy as np
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
from torch.nn.parameter import Parameter

#################################### Utils ###############################
# Gradient Surgery
def project_conflicting(sub_grads, main_grad):
    pc_grad, num_task = copy.deepcopy(sub_grads), len(sub_grads)

    for g_i in pc_grad:
        g_i_g_j = torch.dot(g_i, main_grad)
        if g_i_g_j < 0:
            g_i -= (g_i_g_j) * main_grad / (main_grad.norm() ** 2)

    pc_grad = torch.stack(pc_grad)
    pc_grad = torch.mean(torch.cat((pc_grad, main_grad.reshape(1, -1)), 0), 0)

    return pc_grad

#################################### Classifier ###############################
# Classifier with Softmax
class Softmax_Classifier(nn.Module):
    def __init__(self, du2, du3):
        super(Softmax_Classifier, self).__init__()

        self.hidden_1 = nn.Sequential(nn.Linear(du2, du3), nn.Tanh())
        self.classifier = nn.Sequential(nn.Linear(du3, 2), nn.Softmax(dim=1))

        # Weight Initialization
        self.hidden_1.apply(initialization)
        self.classifier.apply(initialization)

    def forward(self, input, target):
        x = self.hidden_1(input)

        x = self.classifier(x)

        lossClassify = F.cross_entropy(x, target.long())
        return lossClassify

    def predict(self, input):
        x = self.hidden_1(input)

        prob = self.classifier(x)
        prediction = torch.argmax(prob, dim=1)

        return prob, prediction

class DMF(nn.Module):
    def __init__(self, X_modality1, X_modality2, X_modality3, du1, du2, device):
        super(DMF, self).__init__()

        # Device Setting
        self.device = device

        # For Matrix Factorization......
        # Weight Initialization => SVD Initialization
        if X_modality1.shape[0] > du1:
            u, \
            (w_modality11, w_modality12), \
            (w_modality21, w_modality22), \
            (w_modality31, w_modality32) = \
                self.svd_initialization(X_modality1, X_modality2, X_modality3, du1, du2)

            # Modality 1=> modality1 Weight
            self.w_modality11 = Parameter(torch.tensor(w_modality11, dtype=torch.float32))
            self.w_modality12 = Parameter(torch.tensor(w_modality12, dtype=torch.float32))

            # Modality 2=> modality2 Weight
            self.w_modality21 = Parameter(torch.tensor(w_modality21, dtype=torch.float32))
            self.w_modality22 = Parameter(torch.tensor(w_modality22, dtype=torch.float32))

            # Modality 3=> modality3 Weight
            self.w_modality31 = Parameter(torch.tensor(w_modality31, dtype=torch.float32))
            self.w_modality32 = Parameter(torch.tensor(w_modality32, dtype=torch.float32))

            # Common Matrix
            self.u = Parameter(torch.Tensor(u))

        # Weight Initialization => Xaiver
        else:
            # Modality 1=> modality1 Weight
            self.w_modality11 = Parameter(torch.Tensor(np.ones((du1, X_modality1.shape[1]))))
            self.w_modality12 = Parameter(torch.Tensor(np.ones((du2, du1))))

            # Modality 2=> modality2 Weight
            self.w_modality21 = Parameter(torch.Tensor(np.ones((du1, X_modality2.shape[1]))))
            self.w_modality22 = Parameter(torch.Tensor(np.ones((du2, du1))))

            # Modality 3=> modality2 Weight
            self.w_modality31 = Parameter(torch.Tensor(np.ones((du1, X_modality3.shape[1]))))
            self.w_modality32 = Parameter(torch.Tensor(np.ones((du2, du1))))

            # Common Matrix
            self.u = Parameter(torch.Tensor(np.ones((X_modality1.shape[0], du2))))

            # Xaiver Initialization
            init.kaiming_uniform_(self.w_modality11)
            init.kaiming_uniform_(self.w_modality12)
            init.kaiming_uniform_(self.w_modality21)
            init.kaiming_uniform_(self.w_modality22)
            init.kaiming_uniform_(self.w_modality31)
            init.kaiming_uniform_(self.w_modality32)
            init.kaiming_uniform_(self.u)

    # Weight Initialization => SVD
    def svd_initialization(self, X_modality1, X_modality2, X_modality3, du1, du2):
        # Modality 1 => modality1
        # SVD
        u_modality1_svd1, _, v_modality1_svd1 = np.linalg.svd(X_modality1, full_matrices=False)
        u_modality1_svd2, _, v_modality1_svd2 = np.linalg.svd(u_modality1_svd1, full_matrices=False)

        # Weight Initialization => Cut by Dimension of Feature
        w_modality11 = v_modality1_svd1[0:du1, :]
        w_modality12 = u_modality1_svd2[0:du2, 0:du1]

        # Modality 2 => modality2
        # SVD
        u_modality2_svd1, _, v_modality2_svd1 = np.linalg.svd(X_modality2, full_matrices=False)
        u_modality2_svd2, _, v_modality2_svd2 = np.linalg.svd(u_modality2_svd1, full_matrices=False)

        # Weight Initialization => Cut by Dimension of Feature
        w_modality21 = v_modality2_svd1[0:du1, :]
        w_modality22 = u_modality2_svd2[0:du2, 0:du1]

        # Modality 3 => modality3
        # SVD
        u_modality3_svd1, _, v_modality3_svd1 = np.linalg.svd(X_modality3, full_matrices=False)
        u_modality3_svd2, _, v_modality3_svd2 = np.linalg.svd(u_modality3_svd1, full_matrices=False)

        # Weight Initialization => Cut by Dimension of Feature
        w_modality31 = v_modality3_svd1[0:du1, :]
        w_modality32 = u_modality3_svd2[0:du2, 0:du1]

        # Common Matrix
        u = u_modality1_svd2[:, 0:du2]

        return u, (w_modality11, w_modality12), (w_modality21, w_modality22), (w_modality31, w_modality32)

    def forward(self, X_modality1, X_modality2, X_modality3):
        modality1_ = torch.tanh(torch.matmul(self.w_modality12, self.w_modality11))
        modality2_ = torch.tanh(torch.matmul(self.w_modality22, self.w_modality21))
        modality3_ = torch.tanh(torch.matmul(self.w_modality32, self.w_modality31))

        # Modality 1 Loss
        X_modality1_ = torch.matmul(self.u, modality1_)
        X_modality1_loss = F.mse_loss(X_modality1_, torch.tensor(X_modality1, dtype=torch.float32).to(self.device))

        # Modality 2 Loss
        X_modality2_ = torch.matmul(self.u, modality2_)
        X_modality2_loss = F.mse_loss(X_modality2_, torch.tensor(X_modality2, dtype=torch.float32).to(self.device))

        # Modality 3 Loss
        X_modality3_ = torch.matmul(self.u, modality3_)
        X_modality3_loss = F.mse_loss(X_modality3_, torch.tensor(X_modality3, dtype=torch.float32).to(self.device))

        return X_modality1_loss, X_modality2_loss, X_modality3_loss

    def get_param(self):
        individual_param_list = [self.w_modality11, self.w_modality12,
                                 self.w_modality21, self.w_modality22,
                                 self.w_modality31, self.w_modality32]
        common_param = [self.u]

        return individual_param_list, common_param