# Import Library
from Utils import *
from Model import *
import argparse
from tqdm import tqdm

# Option
def parse_args():
    parser = argparse.ArgumentParser(description='MMMF')

    # Device
    parser.add_argument('--device', help='Using GPU Device', type=int)

    # Patience
    parser.add_argument('--patience', help='For Early Stopping', type=int)

    # DataType
    parser.add_argument('--data_type', help='0:ADNI, 1:ROSMAP, 2:BRCA, 3:KIRC, 4:Colon', type=int)

    # Number of Feature Top features
    parser.add_argument('--mp', help='0:1000, 1:2000, 2:3000', type=int)

    # Dimension List
    parser.add_argument('--du_list', help='Dimension List (Hyperparameter)', action='append')

    # Reconstruction Learning Rate
    parser.add_argument('--re_lr', help='Reconstruction Learning Rate', type=float)

    # Reconstruction Regularization
    parser.add_argument('--re_reg', help='Reconstruction Regularization', type=float)

    # Classification Learning Rate
    parser.add_argument('--clf_lr', help='Classification Learning Rate', type=float)

    # Classification Regularization
    parser.add_argument('--clf_reg', help='Classification Regularization', type=float)

    # Patience
    parser.add_argument('--re_patience', help='Reconstruction Initialization', type=int)

    return parser.parse_args()

# Model Train with 3 Modality
def train(dataset, hyper_dict, multi_class_flip, patience, classification_save_root_path):
    # Validation Result
    validation_ba_list = []
    validation_f1_list = []
    validation_auc_list = []
    validation_mcc_list = []

    # Test Result
    test_ba_list = []
    test_f1_list = []
    test_auc_list = []
    test_mcc_list = []

    # 5 Cross Validation
    for cv in range(5):
        # Load Dataset
        [modality1_values, modality2_values, modality3_values], \
        [train_index, validation_index, test_index], \
        [y_train, y_val, y_test] = dataset(cv)

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
        mf_individual_optimizer = torch.optim.Adam(individual_param_list, lr=hyper_dict['re_lr'], weight_decay=hyper_dict['re_reg'])
        mf_common_optimizer = torch.optim.Adam(common_param, lr=hyper_dict['clf_lr'], weight_decay=hyper_dict['clf_reg'])
        clf_optimizer = torch.optim.Adam(clf.parameters(), lr=hyper_dict['clf_lr'], weight_decay=hyper_dict['clf_reg'])

        # Model Train // Validation
        for i in range(hyper_dict['epoch']):
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
            if i > patience:
                smote = SMOTE(random_state=0)
                u_over, y_train_over = smote.fit_resample(net.u[train_index].cpu().detach().numpy(), y_train)

                u_over = torch.tensor(u_over, requires_grad=True).float()
                u_over.retain_grad()

                # Softmax Classifier
                clf_loss = clf(u_over.to(hyper_dict['device']), torch.tensor(y_train_over).to(hyper_dict['device'])).to(hyper_dict['device'])
                clf_loss.backward()

                g4 = torch.zeros_like(net.u.grad).to(hyper_dict['device'])
                g4[train_index] = u_over.grad[:len(train_index)].clone().to(hyper_dict['device'])
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
            clf_loss = clf(net.u[validation_index], torch.tensor(y_val).to(hyper_dict['device']))

            # Check Early Stopping
            early_stopping(clf_loss)
            if clf_loss < best_loss:
                net_best_model_sd = copy.deepcopy(net.state_dict())
                clf_best_model_sd = copy.deepcopy(clf.state_dict())
                best_loss = clf_loss

            if early_stopping.early_stop:
                break

        # Save Model
        net_save_file_name = str(hyper_dict['du1']) + '_' + str(hyper_dict['du2']) + '_' + str(hyper_dict['du3']) + '_net.pt'
        clf_save_file_name = str(hyper_dict['du1']) + '_' + str(hyper_dict['du2']) + '_' + str(hyper_dict['du3']) + '_clf.pt'
        torch.save(net_best_model_sd, os.path.join(classification_save_root_path, 'CV' + str(cv + 1), net_save_file_name))
        torch.save(clf_best_model_sd, os.path.join(classification_save_root_path, 'CV' + str(cv + 1), clf_save_file_name))

        # For Validation & Test Model Performance
        net.load_state_dict(net_best_model_sd)
        net.eval()
        clf.load_state_dict(clf_best_model_sd)
        clf.eval()

        # Validation
        prob, prediction = clf.predict(net.u[validation_index])
        prob = prob.detach().cpu().numpy()
        prediction = prediction.detach().cpu().numpy()
        ba, f1, auc, mcc = calculate_metric(y_val, prediction, prob, multi_class_flip)
        validation_ba_list.append(ba)
        validation_f1_list.append(f1)
        validation_auc_list.append(auc)
        validation_mcc_list.append(mcc)

        # Test
        prob, prediction = clf.predict(net.u[test_index])
        prob = prob.detach().cpu().numpy()
        prediction = prediction.detach().cpu().numpy()
        ba, f1, auc, mcc = calculate_metric(y_test, prediction, prob, multi_class_flip)
        test_ba_list.append(ba)
        test_f1_list.append(f1)
        test_auc_list.append(auc)
        test_mcc_list.append(mcc)

    result = {'Validation': {'BA': validation_ba_list, 'F1': validation_f1_list, 'AUC': validation_auc_list, 'MCC': validation_mcc_list},
              'Test': {'BA': test_ba_list, 'F1': test_f1_list, 'AUC': test_auc_list, 'MCC': test_mcc_list}}

    return result


if __name__ == '__main__':
    # Option Setting
    args = parse_args()

    # Seed Setting
    set_seed()

    # Fix DataType
    data_type_list = ['ADNI', 'ROSMAP', 'BRCA', 'KIRC', 'Colon']
    data_type = data_type_list[args.data_type]

    # Fix mp
    mp_list = [1000, 2000, 3000]
    mp = mp_list[args.mp]

    # BRCA -> Multi-Class
    if data_type == 'BRCA':
        multi_class_flip = True
    else:
        multi_class_flip = False

    # Print Argument
    print('MMMF // Data Type: {} MP: {}'.format(data_type, mp))

    # Save Path
    performance_save_root_path = os.path.join('./Result/Performance/', 'Patience_'+str(args.re_patience), str(mp), data_type, 'MMMF')
    createFolder(performance_save_root_path)

    classification_save_root_path = os.path.join('./Result/Classifcation_Model/', 'Patience_' + str(args.re_patience), str(mp), data_type, 'MMMF')
    for cv in range(5):
        createFolder(os.path.join(classification_save_root_path, 'CV' + str(cv + 1)))

    # Prepare Dataset
    dataset = Dataset(data_type, mp)

    # Prepare Hyperparameter
    hyper_parameters = Hyperparameters('MMMF', args.device, args.patience,
                                       list(map(int, args.du_list)),
                                       args.clf_lr, args.clf_reg, args.re_lr, args.re_reg)
    all_hyper_parameters = hyper_parameters.all_hyper_parameters()
    hyper_parameters_list = hyper_parameters.param_list

    # Grid Search
    for all_hyper_parameter in tqdm(hyper_parameters_list, desc='MMMF Hyperparameter Search...'):
        one_hyper_parameter = hyper_parameters.one_hyper_parameters(all_hyper_parameter)
        result = train(dataset, all_hyper_parameter, multi_class_flip, args.re_patience, classification_save_root_path)
        Result_Save(result, performance_save_root_path, all_hyper_parameters, one_hyper_parameter)