from Utils import *
import argparse
from tqdm import tqdm_notebook as tqdm

# Option
def parse_args():
    parser = argparse.ArgumentParser(description='Pathway')

    # Device
    parser.add_argument('--device', help='Using GPU Device', type=int)

    # Patience
    parser.add_argument('--patience', help='For Early Stopping', type=int)

    # DataType
    parser.add_argument('--data_type', help='0:ADNI, 1:BRCA', type=int)

    return parser.parse_args()


# Model Train with 3 Modality
def train(dataset, hyper_dict, multi_class_flip):
    # Load Dataset
    [modality1_values, modality2_values, modality3_values], y = dataset()

    # For Early Stopping
    early_stopping = EarlyStopping(patience=hyper_dict['patience'], delta=0)
    best_loss = np.Inf
    net_best_model_sd = None
    clf_best_model_sd = None

    # Define Model
    net = DMF(modality1_values, modality2_values, modality3_values,
              hyper_dict['du1'], hyper_dict['du2'], hyper_dict['device']).to(hyper_dict['device'])

    clf = Softmax_Classifier(hyper_dict['du2'], hyper_dict['du3'], multi_class_flip).to(hyper_dict['device'])

    # Optimizer
    individual_param_list, common_param = net.get_param()
    mf_individual_optimizer = torch.optim.Adam(individual_param_list, lr=hyper_dict['re_lr'],
                                               weight_decay=hyper_dict['re_reg'])
    mf_common_optimizer = torch.optim.Adam(common_param, lr=hyper_dict['clf_lr'], weight_decay=hyper_dict['clf_reg'])
    clf_optimizer = torch.optim.Adam(clf.parameters(), lr=hyper_dict['clf_lr'], weight_decay=hyper_dict['clf_reg'])

    # Model Train // Validation
    for i in tqdm(range(hyper_dict['epoch']), desc='Model Training...'):
        # For Train
        net.train()
        clf.train()

        # Optimzer.zero_grad()
        mf_individual_optimizer.zero_grad()
        mf_common_optimizer.zero_grad()
        clf_optimizer.zero_grad()

        # Reconstruction
        m1_loss, m2_loss, m3_loss = net(modality1_values, modality2_values, modality3_values)

        # Reconstruction M1 Grad of U
        m1_loss.backward()
        g1 = net.u.grad.clone()
        net.u.grad.zero_()

        # Reconstruction M2 Grad of U
        m2_loss.backward()
        g2 = net.u.grad.clone()
        net.u.grad.zero_()

        # Reconstruction M3 Grad of U
        m3_loss.backward()
        g3 = net.u.grad.clone()
        net.u.grad.zero_()

        # Reconstruction CLF Grad of U
        # Over Sampling
        if i > hyper_dict['re_patience']:
            smote = SMOTE(random_state=0)
            u_over, y_over = smote.fit_resample(net.u.cpu().detach().numpy(), y)

            u_over = torch.tensor(u_over, requires_grad=True).float()
            u_over.retain_grad()

            # Softmax Classifier
            clf_loss = clf(u_over.to(hyper_dict['device']),
                           torch.tensor(y_over).to(hyper_dict['device'])).to(hyper_dict['device'])
            clf_loss.backward()

            g4 = torch.zeros_like(net.u.grad).to(hyper_dict['device'])
            g4 = u_over.grad[:len(y)].clone().to(hyper_dict['device'])
            net.u.grad.zero_()

            # Projection Grad => U[train_index] -> g1, g2, g3, g4, U[~train_index] -> g1, g2, g3
            proj_grad = project_conflicting([g1.flatten(), g2.flatten(), g3.flatten()], g4.flatten())
            proj_grad = proj_grad.reshape_as(net.u.grad)
            net.u.grad = proj_grad

            # optimizer.step()
            mf_individual_optimizer.step()
            mf_common_optimizer.step()
            clf_optimizer.step()

        else:
            mean_grad = (g1 + g2 + g3) / 3
            net.u.grad = mean_grad
            # optimizer.step()
            mf_individual_optimizer.step()
            mf_common_optimizer.step()

        # Model Validation
        net.eval()
        clf.eval()
        clf_loss = clf(net.u, torch.tensor(y).to(hyper_dict['device']))
        m1_loss, m2_loss, m3_loss = net(modality1_values, modality2_values, modality3_values)

        loss = clf_loss + m1_loss + m2_loss + m3_loss

        # Check Early Stopping
        early_stopping(loss)
        if loss < best_loss:
            net_best_model_sd = copy.deepcopy(net.state_dict())
            clf_best_model_sd = copy.deepcopy(clf.state_dict())
            best_loss = loss

        if early_stopping.early_stop:
            break

    # Save Model
    net_save_path = os.path.join('./Result/Pathway_Model/', data_type)
    clf_save_path = os.path.join('./Result/Pathway_Model/', data_type)
    createFolder(net_save_path)
    createFolder(clf_save_path)
    net_save_file_name = str(hyper_dict['du1']) + '_' + str(hyper_dict['du2']) + '_' + str(
        hyper_dict['du3']) + '_net.pt'
    clf_save_file_name = str(hyper_dict['du1']) + '_' + str(hyper_dict['du2']) + '_' + str(
        hyper_dict['du3']) + '_clf.pt'
    torch.save(net_best_model_sd, os.path.join(net_save_path, net_save_file_name))
    torch.save(clf_best_model_sd, os.path.join(clf_save_path, clf_save_file_name))

    return None

if __name__ == '__main__':
    # Option Setting
    args = parse_args()

    # Seed Setting
    set_seed()

    # Device
    device = torch.device("cuda:" + str(args.device) if torch.cuda.is_available() else "cpu")

    # Fix DataType
    data_type_list = ['ADNI', 'BRCA']
    data_type = data_type_list[args.data_type]

    # BRCA -> Multi-Class
    if data_type == 'BRCA':
        multi_class_flip = True
    else:
        multi_class_flip = False

    # Print Argument
    print('Pathway // Data Type: {}'.format(data_type))

    # Prepare Dataset
    dataset = All_Dataset(data_type)

    # Both ADNI and BRCA datasets have the same hyperparameters when the best performance is achieved.
    du1, du2, du3, re_patience = 110, 90, 70, 50
    # 50000
    hyper_dict = {'patience': args.patience, 'epoch': 2,
                  'device': device,
                  'du1': du1, 'du2': du2, 'du3': du3,
                  'clf_lr': 0.001, 'clf_reg': 0.0001, 're_lr': 0.0001, 're_reg': 0.0001, 're_patience': re_patience}

    train(dataset, hyper_dict, multi_class_flip)
