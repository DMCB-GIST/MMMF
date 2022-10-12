# Import Library
from Utils import *
import os
import argparse
from tqdm import tqdm
from sklearn.svm import SVC

# Option
def parse_args():
    parser = argparse.ArgumentParser(description='Feature Selection using SVM')

    # Device
    parser.add_argument('--device', help='Using GPU Device', type=int)

    # DataType
    parser.add_argument('--data_type', help='0:ADNI, 1:ROSMAP, 2:BRCA, 3:KIRC, 4:Colon', type=int)

    # Number of Feature Top features
    parser.add_argument('--mp', help='0:1000, 1:2000, 2:3000', type=int)

    # Top n
    parser.add_argument('--top_n', help='0:5, 1:10, 2:20, 3:30', type=int)

    # Feature Selection Method
    parser.add_argument('--feature_selection', help='0: MMMF, 1: Random', type=int)

    return parser.parse_args()

# Model Training
def train(dataset, one_hyper_parameter, multi_class_flip):
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

    # Outer CV = 5
    for cv in range(5):
        # Load Dataset
        [x_train, x_val, x_test], [y_train, y_val, y_test] = dataset(cv)

        clf = SVC(**one_hyper_parameter)
        clf.fit(x_train, y_train)

        # SVM Validation Performance
        y_pred = clf.predict(x_val)
        y_pred_proba = clf.predict_proba(x_val)
        ba, f1, auc, mcc = calculate_metric(y_val, y_pred, y_pred_proba, multi_class_flip)
        validation_ba_list.append(ba)
        validation_f1_list.append(f1)
        validation_auc_list.append(auc)
        validation_mcc_list.append(mcc)

        # SVM Test Performance
        y_pred = clf.predict(x_test)
        y_pred_proba = clf.predict_proba(x_test)
        ba, f1, auc, mcc = calculate_metric(y_test, y_pred, y_pred_proba, multi_class_flip)
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

    # Device
    device = torch.device("cuda:" + str(args.device) if torch.cuda.is_available() else "cpu")

    # Fix DataType
    data_type_list = ['ADNI', 'ROSMAP', 'BRCA', 'KIRC', 'Colon']
    data_type = data_type_list[args.data_type]

    # Fix mp
    mp_list = [1000, 2000, 3000]
    mp = mp_list[args.mp]

    # Fix top_n
    top_n_list = [5, 10, 20, 30]
    top_n = top_n_list[args.top_n]

    # BRCA -> Multi-Class
    if data_type == 'BRCA':
        multi_class_flip = True
    else:
        multi_class_flip = False

    # Feature Selection Method
    feature_selection_list = ['MMMF', 'Random']
    feature_selection = feature_selection_list[args.feature_selection]

    # Print Argument
    print('SVM // Data Type: {}, MP: {}, Top n: {}, Feature Selection Method: {}'.format(data_type, mp, top_n, feature_selection))

    # Make Feature Selection Data
    if feature_selection == 'MMMF':
        net_list = check_classification_performance(data_type, mp, 'MMMF', device)
        make_feature_selection_data(net_list, data_type, mp, top_n=top_n)

    # Prepare Hyperparameter
    hyper_parameters = Hyperparameters('SVM')
    all_hyper_parameters = hyper_parameters.all_hyper_parameters()
    hyper_parameters_list = hyper_parameters.param_list

    number_of_random_experiment = 100

    # Grid Search
    for all_hyper_parameter in tqdm(hyper_parameters_list, desc='SVM Hyperparameter Search...'):
        one_hyper_parameter = hyper_parameters.one_hyper_parameters(all_hyper_parameter)

        # Random Selection
        if feature_selection == 'Random':
            # Save Path
            save_root_path = os.path.join('./Result/Feature_Selection_Random/', str(top_n), str(mp), data_type)
            createFolder(save_root_path)

            for random_state in range(number_of_random_experiment):
                # Prepare Dataset
                dataset = FS_Dataset(data_type, mp, top_n, feature_selection, random_state)
                result = train(dataset, all_hyper_parameter, multi_class_flip)
                Random_Result_Save(result, save_root_path, all_hyper_parameters, one_hyper_parameter, random_state)

        # Feature Selection with MMMF
        else:
            # Save Path
            save_root_path = os.path.join('./Result/Feature_Selection/', str(top_n), str(mp), data_type)
            createFolder(save_root_path)

            # Prepare Dataset
            dataset = FS_Dataset(data_type, mp, top_n, feature_selection)
            result = train(dataset, all_hyper_parameter, multi_class_flip)
            Result_Save(result, save_root_path, all_hyper_parameters, one_hyper_parameter)
