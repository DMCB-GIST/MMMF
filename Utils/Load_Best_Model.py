from Model import *
from Utils.Data import *
from Utils.ETC import calculate_metric
from sklearn.svm import SVC

# Load Best Model
def check_classification_performance(data_type, mp, model, device):
    net_list = []

    ba_list = np.zeros(5)
    f1_list = np.zeros(5)
    auc_list = np.zeros(5)
    mcc_list = np.zeros(5)

    dataset = Dataset(data_type, mp)
    model_root_path = os.path.join('./Result/Model/Classification/', str(mp), data_type, model)

    # Load MMMF-Over-GS, MMMF-Over, MMMF-GS, MMMF
    for cv in range(5):
        if data_type != 'BRCA':
            multi_class_flip = False
        else:
            multi_class_flip = True

        # Load Dataset
        # Prepare Dataset
        [modality1_values, modality2_values, modality3_values], [_, _, test_index], \
        [_, _, y_test] = dataset(cv)

        # Load Best Hyperparamters & Model
        for f in os.listdir(os.path.join(model_root_path, 'CV' + str(cv + 1))):
            du1 = int(f.split('_')[0])
            du2 = int(f.split('_')[1])
            du3 = int(f.split('_')[2])

            if 'net.pt' in f:
                net_path = os.path.join(model_root_path, 'CV' + str(cv + 1), f)
            elif 'clf.pt' in f:
                clf_path = os.path.join(model_root_path, 'CV' + str(cv + 1), f)

        net = DMF(modality1_values, modality2_values, modality3_values, du1, du2, device).to(device)
        clf = Softmax_Classifier(du2, du3, multi_class_flip).to(device)

        net.load_state_dict(torch.load(net_path, map_location=device))
        clf.load_state_dict(torch.load(clf_path, map_location=device))

        net.eval()
        clf.eval()

        net_list.append(net)

        # Test
        prob, prediction = clf.predict(net.u[test_index])
        prob = prob.detach().cpu().numpy()
        prediction = prediction.detach().cpu().numpy()
        ba, f1, auc, mcc = calculate_metric(y_test, prediction, prob, multi_class_flip)

        ba_list[cv] = ba
        f1_list[cv] = f1
        auc_list[cv] = auc
        mcc_list[cv] = mcc

    print('Reproduction')
    print('Test BA: {:.3f}±{:.3f}, F1: {:.3f}±{:.3f}, '
          'AUC: {:.3f}±{:.3f}, MCC: {:.3f}±{:.3f}'.format(ba_list.mean(), ba_list.std(), f1_list.mean(), f1_list.std(),
                                                          auc_list.mean(), auc_list.std(), mcc_list.mean(),
                                                          mcc_list.std()))

    return net_list

def check_feature_selection_performance(data_type, mp, top_n):
    # Load Best Hyperparmeter
    best_hyper_list = []
    result_root_path = os.path.join('./Result/Feature_Selection', str(top_n), str(mp), data_type)
    validation = pd.read_csv(os.path.join(result_root_path, 'Validation.csv'), index_col=0)
    test = pd.read_csv(os.path.join(result_root_path, 'Test.csv'), index_col=0)

    num_hyper_params = validation.shape[0]
    validation_metric = 'AUC'
    metric_list = ['BA', 'F1', 'AUC', 'MCC']
    metric_index = metric_list.index(validation_metric)

    # Search by Validation Performance
    val_result = np.zeros((num_hyper_params, 5))
    for k in range(num_hyper_params):
        # 5 CV
        for i in range(5):
            val_result[k, i] = float(validation.iloc[k, metric_index].split(',')[i])

    index = np.argmax(val_result, 0)
    ba = np.zeros(5)
    f1 = np.zeros(5)
    auc = np.zeros(5)
    mcc = np.zeros(5)

    for i in range(5):
        ba[i] = float(test.iloc[index[i], 0].split(',')[i])
        f1[i] = float(test.iloc[index[i], 1].split(',')[i])
        auc[i] = float(test.iloc[index[i], 2].split(',')[i])
        mcc[i] = float(test.iloc[index[i], 3].split(',')[i])

        best_hyper_list.append({'gamma': 'scale', 'random_state': 3, 'probability': True,
                                'C': int(test.index[index[i]].split('_')[0]),
                                'kernel': test.index[index[i]].split('_')[1]})
    print('Check Best Hyperparmeter Performance')
    print('Test BA: {:.3f}±{:.3f}, F1: {:.3f}±{:.3f}, AUC: {:.3f}±{:.3f}, MCC: {:.3f}±{:.3f}'.format(
        np.array(ba).mean(), np.array(ba).std(),
        np.array(f1).mean(), np.array(f1).std(),
        np.array(auc).mean(), np.array(auc).std(),
        np.array(mcc).mean(), np.array(mcc).std()))

    # Test Result
    dataset = FS_Dataset(data_type, mp, top_n, feature_selection='MMMF')
    test_ba_list = []
    test_f1_list = []
    test_auc_list = []
    test_mcc_list = []

    # Outer CV = 5
    for cv in range(5):
        # Check Dataset Probability
        if data_type != 'BRCA':
            multi_class_flip = False
        else:
            multi_class_flip = True

        # Load Dataset
        [x_train, _, x_test], [y_train, _, y_test] = dataset(cv)

        clf = SVC(**best_hyper_list[cv])
        clf.fit(x_train, y_train)

        # SVM Test Performance
        y_pred = clf.predict(x_test)
        y_pred_proba = clf.predict_proba(x_test)
        ba, f1, auc, mcc = calculate_metric(y_test, y_pred, y_pred_proba, multi_class_flip)
        test_ba_list.append(ba)
        test_f1_list.append(f1)
        test_auc_list.append(auc)
        test_mcc_list.append(mcc)

    print('Reproduction')
    print('Test BA: {:.3f}±{:.3f}, F1: {:.3f}±{:.3f}, AUC: {:.3f}±{:.3f}, MCC: {:.3f}±{:.3f}'.format(
        np.array(test_ba_list).mean(), np.array(test_ba_list).std(),
        np.array(test_f1_list).mean(), np.array(test_f1_list).std(),
        np.array(test_auc_list).mean(), np.array(test_auc_list).std(),
        np.array(test_mcc_list).mean(), np.array(test_mcc_list).std()))