from scipy import stats
from Model import *
from Utils.Data import *
from Utils.ETC import createFolder

def make_feature_selection_data(net_list, data_type, mp, top_n=5):
    p_val_threshold = 0.05

    dataset = Dataset(data_type, mp)

    if data_type == 'ADNI':
        m1_name, m2_name, m3_name = 'smri', 'pet', 'gene'
    elif data_type == 'ROSMAP':
        m1_name, m2_name, m3_name = 'GE', 'ME', 'MI'
    elif data_type == 'BRCA':
        m1_name, m2_name, m3_name = 'gene', 'prot', 'cnv'
    elif data_type == 'KIRC':
        m1_name, m2_name, m3_name = 'GE', 'ME', 'MI'
    else:
        m1_name, m2_name, m3_name = 'GE', 'MI', 'CpGs'

    print(mp, data_type)

    # Feature Selection by Model
    for cv in range(5):
        # Make Directory
        save_root_path = os.path.join('./Result/Feature_Selection_Data/', str(top_n), str(mp), data_type)
        createFolder(save_root_path)
        net = net_list[cv]

        # Prepare Dataset
        _, [_, validation_index, _], [_, y_val, _] = dataset(cv)

        data_index = validation_index
        label = y_val
        data = net.u[data_index]

        # Binary -> T-test
        if data_type != 'BRCA':
            class_0_data = data[label == 0].cpu().detach().numpy()
            class_1_data = data[label == 1].cpu().detach().numpy()
            t, p = stats.ttest_ind(class_0_data, class_1_data)

        # Multi-Class
        else:
            class_0_data = data[label == 0].cpu().detach().numpy()
            class_1_data = data[label == 1].cpu().detach().numpy()
            class_2_data = data[label == 2].cpu().detach().numpy()
            class_3_data = data[label == 3].cpu().detach().numpy()
            F, p = stats.f_oneway(class_0_data, class_1_data, class_2_data, class_3_data)

        # P Value -> Min -> Significant Module
        selected_module = np.arange(len(p))[p < p_val_threshold]
        print('Number of Significant Module: {}'.format(len(selected_module)))
        if len(selected_module) == 0:
            print('Find No Significant Module --> Select Minimum P-value Module')
            selected_module = np.array([np.argsort(p)[0]])
            print('Number of Significant Module: {}'.format(len(selected_module)))

        # Modality 1 Sorting Feature Importance & Save
        modality_1_individual = torch.tanh(torch.matmul(net.w_modality12, net.w_modality11)).cpu().detach().numpy()

        # Coefficient -> Z Score -> P Value
        modality_1_feature_coef = np.abs(modality_1_individual[selected_module, :])
        modality_1_selected_feature_index = []
        for i in range(len(selected_module)):
            modality1_f = modality_1_feature_coef[i, :]
            modality_1_selected_feature_index += list(np.argsort(modality1_f)[::-1][:top_n])
        modality_1_selected_feature_index = list(set(modality_1_selected_feature_index))

        modality_1_selected_feature = list(np.array(dataset.modality1.columns.to_list()[2:])[modality_1_selected_feature_index])
        dataset.modality1[['Subject', 'Label'] + modality_1_selected_feature].to_csv(os.path.join(save_root_path, 'cv' + str(cv + 1) + m1_name + '.csv'))

        # Modality 2 Sorting Feature Importance & Save
        modality_2_individual = torch.tanh(torch.matmul(net.w_modality22, net.w_modality21)).cpu().detach().numpy()

        # Coefficient -> Z Score -> P Value
        modality_2_feature_coef = np.abs(modality_2_individual[selected_module, :])
        modality_2_selected_feature_index = []
        for i in range(len(selected_module)):
            modality2_f = modality_2_feature_coef[i, :]
            modality_2_selected_feature_index += list(np.argsort(modality2_f)[::-1][:top_n])
        modality_2_selected_feature_index = list(set(modality_2_selected_feature_index))

        modality_2_selected_feature = list(np.array(dataset.modality2.columns.to_list()[2:])[modality_2_selected_feature_index])
        dataset.modality2[['Subject', 'Label'] + modality_2_selected_feature].to_csv(os.path.join(save_root_path, 'cv' + str(cv + 1) + m2_name + '.csv'))

        modality_3_individual = torch.tanh(torch.matmul(net.w_modality32, net.w_modality31)).cpu().detach().numpy()

        # Coefficient -> Z Score -> P Value
        modality_3_feature_coef = np.abs(modality_3_individual[selected_module, :])
        modality_3_selected_feature_index = []
        for i in range(len(selected_module)):
            modality3_f = modality_3_feature_coef[i, :]
            modality_3_selected_feature_index += list(np.argsort(modality3_f)[::-1][:top_n])
        modality_3_selected_feature_index = list(set(modality_3_selected_feature_index))

        modality_3_selected_feature = list(np.array(dataset.modality3.columns.to_list()[2:])[modality_3_selected_feature_index])
        dataset.modality3[['Subject', 'Label'] + modality_3_selected_feature].to_csv(os.path.join(save_root_path, 'cv' + str(cv + 1) + m3_name + '.csv'))