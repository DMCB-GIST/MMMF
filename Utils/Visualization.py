from numpy import interp
from Utils import *
import matplotlib.pyplot as plt
from sklearn.metrics import auc
from sklearn.metrics import roc_curve, precision_recall_curve



def classification_roc_curve_plot(ax, data_type, mp, model_list):
    for model in model_list:
        file_root_path = os.path.join('./Result/Probability/Classification/', str(mp), data_type, model)

        tprs = []
        aucs = []
        mean_fpr = np.linspace(0, 1, 10000)

        for cv in range(5):
            file_path = os.path.join('CV' + str(cv + 1), 'Probability.csv')
            result_pd = pd.read_csv(os.path.join(file_root_path, file_path), index_col=0)

            y_test = result_pd['Label'].values
            y_predict_prob = result_pd['Pro-1'].values
            fpr, tpr, thresholds = roc_curve(y_test, y_predict_prob)
            roc_auc = auc(fpr, tpr)

            tprs.append(interp(mean_fpr, fpr, tpr))

            tprs[-1][0] = 0.0
            roc_auc = auc(fpr, tpr)
            aucs.append(roc_auc)

        mean_tpr = np.mean(tprs, axis=0)
        mean_tpr[-1] = 1.0
        mean_auc = auc(mean_fpr, mean_tpr)
        std_auc = np.std(aucs)
        ax.plot(mean_fpr, mean_tpr, label=r'' + model + ' (AUC = %0.3f$\pm$%0.3f)' % (mean_auc, std_auc), lw=4, alpha=.8)

        plt.xlim([-0.01, 1.01])
        plt.ylim([-0.01, 1.01])

        plt.xticks(fontsize=30)
        plt.yticks(fontsize=30)

    plt.title(r'$m_p$: {}'.format(mp), fontsize=60)
    plt.legend(loc="lower right", prop={'size': 24})

    return ax


def classification_pr_curve_plot(ax, data_type, mp, model_list):
    for model in model_list:
        file_root_path = os.path.join('./Result/Probability/Classification/', str(mp), data_type, model)

        prs = []
        aucs = []
        mean_recall = np.linspace(0, 1, 10000)

        for cv in range(5):
            file_path = os.path.join('CV' + str(cv + 1), 'Probability.csv')
            result_pd = pd.read_csv(os.path.join(file_root_path, file_path), index_col=0)

            y_test = result_pd['Label'].values
            y_predict_prob = result_pd['Pro-1'].values
            precision, recall, thresholds = precision_recall_curve(y_test, y_predict_prob)
            prs.append(interp(mean_recall, precision, recall))
            pr_auc = auc(recall, precision)
            aucs.append(pr_auc)

        mean_precision = np.mean(prs, axis=0)
        mean_auc = auc(mean_recall, mean_precision)
        std_auc = np.std(aucs)

        ax.plot(mean_precision, mean_recall, label=r'' + model + ' (PR AUC = %0.3f$\pm$%0.3f)' % (mean_auc, std_auc), lw=4, alpha=.8)

        plt.xlim([-0.01, 1.01])
        plt.ylim([-0.01, 1.01])

        plt.xticks(fontsize=30)
        plt.yticks(fontsize=30)

    plt.title(r'$m_p$: {}'.format(mp), fontsize=60)
    plt.legend(loc="lower left", prop={'size': 24})

    return ax

def classification_roc_curve_plot2(ax, data_type, mp, model_list, class_0, class_1):
    if class_0 == 0:
        model_name_1 = 'Luminal A'
    elif class_0 == 1:
        model_name_1 = 'Luminal B'
    elif class_0 == 2:
        model_name_1 = 'Basal-like'
    else:
        model_name_1 = 'HER2-enriched'

    if class_1 == 0:
        model_name_2 = 'Luminal A'
    elif class_1 == 1:
        model_name_2 = 'Luminal B'
    elif class_1 == 2:
        model_name_2 = 'Basal-like'
    else:
        model_name_2 = 'HER2-enriched'

    for model in model_list:
        file_root_path = os.path.join('./Result/Probability/Classification/', str(mp), data_type, model)

        tprs = []
        aucs = []
        mean_fpr = np.linspace(0, 1, 10000)

        for cv in range(5):
            file_path = os.path.join('CV' + str(cv + 1), 'Probability.csv')
            result_pd = pd.read_csv(os.path.join(file_root_path, file_path), index_col=0)

            result_pd = result_pd[(result_pd['Label'] == class_0) | (result_pd['Label'] == class_1)].copy()
            result_pd['Label'] = result_pd['Label'].apply(lambda x: 1 if (x == class_0) else 0)
            result_pd = result_pd.reset_index(drop=True)

            y_test = result_pd['Label'].values
            y_predict_prob = result_pd['Pro-' + str(class_0)].values

            fpr, tpr, thresholds = roc_curve(y_test, y_predict_prob)
            roc_auc = auc(fpr, tpr)

            tprs.append(interp(mean_fpr, fpr, tpr))

            tprs[-1][0] = 0.0
            roc_auc = auc(fpr, tpr)
            aucs.append(roc_auc)

        mean_tpr = np.mean(tprs, axis=0)
        mean_tpr[-1] = 1.0
        mean_auc = auc(mean_fpr, mean_tpr)
        std_auc = np.std(aucs)
        ax.plot(mean_fpr, mean_tpr,
                label=r'' + model + ' (AUC = %0.3f$\pm$%0.3f)' % (mean_auc, std_auc),
                lw=4, alpha=.8)

        plt.xlim([-0.01, 1.01])
        plt.ylim([-0.01, 1.01])

        plt.xticks(fontsize=30)
        plt.yticks(fontsize=30)

    title = '\nClass 0: {}, Class 1: {}'.format(model_name_2, model_name_1)

    plt.title(title, fontsize=50)
    plt.legend(loc="lower right", prop={'size': 24})

    return ax

def classification_pr_curve_plot2(ax, data_type, mp, model_list, class_0, class_1):
    if class_0 == 0:
        model_name_1 = 'Luminal A'
    elif class_0 == 1:
        model_name_1 = 'Luminal B'
    elif class_0 == 2:
        model_name_1 = 'Basal-like'
    else:
        model_name_1 = 'HER2-enriched'

    if class_1 == 0:
        model_name_2 = 'Luminal A'
    elif class_1 == 1:
        model_name_2 = 'Luminal B'
    elif class_1 == 2:
        model_name_2 = 'Basal-like'
    else:
        model_name_2 = 'HER2-enriched'

    for model in model_list:
        file_root_path = os.path.join('./Result/Probability/Classification/', str(mp), data_type, model)

        prs = []
        aucs = []
        mean_recall = np.linspace(0, 1, 10000)

        for cv in range(5):
            file_path = os.path.join('CV' + str(cv + 1), 'Probability.csv')
            result_pd = pd.read_csv(os.path.join(file_root_path, file_path), index_col=0)

            result_pd = result_pd[(result_pd['Label'] == class_0) | (result_pd['Label'] == class_1)].copy()
            result_pd['Label'] = result_pd['Label'].apply(lambda x: 1 if (x == class_0) else 0)
            result_pd = result_pd.reset_index(drop=True)

            y_test = result_pd['Label'].values
            y_predict_prob = result_pd['Pro-' + str(class_0)].values

            precision, recall, thresholds = precision_recall_curve(y_test, y_predict_prob)
            prs.append(interp(mean_recall, precision, recall))
            pr_auc = auc(recall, precision)
            aucs.append(pr_auc)

        mean_precision = np.mean(prs, axis=0)
        mean_auc = auc(mean_recall, mean_precision)
        std_auc = np.std(aucs)

        ax.plot(mean_precision, mean_recall,
                label=r'' + model + ' (PR AUC = %0.3f$\pm$%0.3f)' % (mean_auc, std_auc),
                lw=4, alpha=.8)

        plt.xlim([-0.01, 1.01])
        plt.ylim([-0.01, 1.01])

        plt.xticks(fontsize=30)
        plt.yticks(fontsize=30)

    title = '\nClass 0: {}, Class 1: {}'.format(model_name_2, model_name_1)

    plt.title(title, fontsize=50)
    plt.legend(loc="lower left", prop={'size': 24})

    return ax

def feature_selection_roc_curve_plot(ax, data_type, mp, model_list):
    for i, model in enumerate(model_list):
        if i == 0:
            file_root_path = os.path.join('./Result/Probability/Classification/', str(mp), data_type, model)
        else:
            file_root_path = os.path.join('./Result/Probability/Feature_Selection/', model, str(mp), data_type)

        tprs = []
        aucs = []
        mean_fpr = np.linspace(0, 1, 10000)

        for cv in range(5):
            file_path = os.path.join('CV' + str(cv + 1), 'Probability.csv')
            result_pd = pd.read_csv(os.path.join(file_root_path, file_path), index_col=0)

            y_test = result_pd['Label'].values
            y_predict_prob = result_pd['Pro-1'].values
            fpr, tpr, thresholds = roc_curve(y_test, y_predict_prob)
            roc_auc = auc(fpr, tpr)

            tprs.append(interp(mean_fpr, fpr, tpr))

            tprs[-1][0] = 0.0
            roc_auc = auc(fpr, tpr)
            aucs.append(roc_auc)

        if model == 'SVM':
            model = 'All'
        elif model == '5':
            model = 'MMMF Top 5'
        elif model == '10':
            model = 'MMMF Top 10'
        elif model == '20':
            model = 'MMMF Top 20'
        elif model == '30':
            model = 'MMMF Top 30'
        else:
            pass

        mean_tpr = np.mean(tprs, axis=0)
        mean_tpr[-1] = 1.0
        mean_auc = auc(mean_fpr, mean_tpr)
        std_auc = np.std(aucs)
        ax.plot(mean_fpr, mean_tpr, label=r'' + model + ' (AUC = %0.3f$\pm$%0.3f)' % (mean_auc, std_auc), lw=4, alpha=.8)

        plt.xlim([-0.01, 1.01])
        plt.ylim([-0.01, 1.01])

        plt.xticks(fontsize=30)
        plt.yticks(fontsize=30)

    plt.title(r'$m_p$: {}'.format(mp), fontsize=60)
    plt.legend(loc="lower right", prop={'size': 24})

    return ax

def feature_selection_pr_curve_plot(ax, data_type, mp, model_list):
    for i, model in enumerate(model_list):
        if i == 0:
            file_root_path = os.path.join('./Result/Probability/Classification/', str(mp), data_type, model)
        else:
            file_root_path = os.path.join('./Result/Probability/Feature_Selection/', model, str(mp), data_type)

        prs = []
        aucs = []
        mean_recall = np.linspace(0, 1, 10000)

        for cv in range(5):
            file_path = os.path.join('CV' + str(cv + 1), 'Probability.csv')
            result_pd = pd.read_csv(os.path.join(file_root_path, file_path), index_col=0)

            y_test = result_pd['Label'].values
            y_predict_prob = result_pd['Pro-1'].values
            precision, recall, thresholds = precision_recall_curve(y_test, y_predict_prob)
            prs.append(interp(mean_recall, precision, recall))
            pr_auc = auc(recall, precision)
            aucs.append(pr_auc)

        if model == 'SVM':
            model = 'All'
        elif model == '5':
            model = 'MMMF Top 5'
        elif model == '10':
            model = 'MMMF Top 10'
        elif model == '20':
            model = 'MMMF Top 20'
        elif model == '30':
            model = 'MMMF Top 30'
        else:
            pass

        mean_precision = np.mean(prs, axis=0)
        mean_auc = auc(mean_recall, mean_precision)
        std_auc = np.std(aucs)

        ax.plot(mean_precision, mean_recall, label=r'' + model + ' (PR AUC = %0.3f$\pm$%0.3f)' % (mean_auc, std_auc), lw=4, alpha=.8)

        plt.xlim([-0.01, 1.01])
        plt.ylim([-0.01, 1.01])

        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)

    plt.title(r'$m_p$: {}'.format(mp), fontsize=60)
    plt.legend(loc="lower left", prop={'size': 24})

    return ax

def feature_selection_roc_curve_plot2(ax, data_type, mp, model_list, class_0, class_1):
    if class_0 == 0:
        model_name_1 = 'Luminal A'
    elif class_0 == 1:
        model_name_1 = 'Luminal B'
    elif class_0 == 2:
        model_name_1 = 'Basal-like'
    else:
        model_name_1 = 'HER2-enriched'

    if class_1 == 0:
        model_name_2 = 'Luminal A'
    elif class_1 == 1:
        model_name_2 = 'Luminal B'
    elif class_1 == 2:
        model_name_2 = 'Basal-like'
    else:
        model_name_2 = 'HER2-enriched'

    for i, model in enumerate(model_list):
        if i == 0:
            file_root_path = os.path.join('./Result/Probability/Classification/', str(mp), data_type, model)
        else:
            file_root_path = os.path.join('./Result/Probability/Feature_Selection/', model, str(mp), data_type)

        tprs = []
        aucs = []
        mean_fpr = np.linspace(0, 1, 10000)

        for cv in range(5):
            file_path = os.path.join('CV' + str(cv + 1), 'Probability.csv')
            result_pd = pd.read_csv(os.path.join(file_root_path, file_path), index_col=0)

            result_pd = result_pd[(result_pd['Label'] == class_0) | (result_pd['Label'] == class_1)].copy()
            result_pd['Label'] = result_pd['Label'].apply(lambda x: 1 if (x == class_0) else 0)
            result_pd = result_pd.reset_index(drop=True)

            y_test = result_pd['Label'].values
            y_predict_prob = result_pd['Pro-' + str(class_0)].values

            fpr, tpr, thresholds = roc_curve(y_test, y_predict_prob)
            roc_auc = auc(fpr, tpr)

            tprs.append(interp(mean_fpr, fpr, tpr))

            tprs[-1][0] = 0.0
            roc_auc = auc(fpr, tpr)
            aucs.append(roc_auc)

        if model == 'SVM':
            model = 'All'
        elif model == '5':
            model = 'MMMF Top 5'
        elif model == '10':
            model = 'MMMF Top 10'
        elif model == '20':
            model = 'MMMF Top 20'
        elif model == '30':
            model = 'MMMF Top 30'
        else:
            pass

        mean_tpr = np.mean(tprs, axis=0)
        mean_tpr[-1] = 1.0
        mean_auc = auc(mean_fpr, mean_tpr)
        std_auc = np.std(aucs)
        ax.plot(mean_fpr, mean_tpr, label=r'' + model + ' (AUC = %0.3f$\pm$%0.3f)' % (mean_auc, std_auc), lw=4, alpha=.8)

        plt.xlim([-0.01, 1.01])
        plt.ylim([-0.01, 1.01])

        plt.xticks(fontsize=30)
        plt.yticks(fontsize=30)

    title = '\nClass 0: {}, Class 1: {}'.format(model_name_2, model_name_1)

    plt.title(title, fontsize=50)
    plt.legend(loc="lower right", prop={'size': 24})

    return ax

def feature_selection_pr_curve_plot2(ax, data_type, mp, model_list, class_0, class_1):
    if class_0 == 0:
        model_name_1 = 'Luminal A'
    elif class_0 == 1:
        model_name_1 = 'Luminal B'
    elif class_0 == 2:
        model_name_1 = 'Basal-like'
    else:
        model_name_1 = 'HER2-enriched'

    if class_1 == 0:
        model_name_2 = 'Luminal A'
    elif class_1 == 1:
        model_name_2 = 'Luminal B'
    elif class_1 == 2:
        model_name_2 = 'Basal-like'
    else:
        model_name_2 = 'HER2-enriched'

    for i, model in enumerate(model_list):
        if i == 0:
            file_root_path = os.path.join('./Result/Probability/Classification/', str(mp), data_type, model)
        else:
            file_root_path = os.path.join('./Result/Probability/Feature_Selection/', model, str(mp), data_type)

        prs = []
        aucs = []
        mean_recall = np.linspace(0, 1, 10000)

        for cv in range(5):
            file_path = os.path.join('CV' + str(cv + 1), 'Probability.csv')
            result_pd = pd.read_csv(os.path.join(file_root_path, file_path), index_col=0)

            result_pd = result_pd[(result_pd['Label'] == class_0) | (result_pd['Label'] == class_1)].copy()
            result_pd['Label'] = result_pd['Label'].apply(lambda x: 1 if (x == class_0) else 0)
            result_pd = result_pd.reset_index(drop=True)

            y_test = result_pd['Label'].values
            y_predict_prob = result_pd['Pro-' + str(class_0)].values

            precision, recall, thresholds = precision_recall_curve(y_test, y_predict_prob)
            prs.append(interp(mean_recall, precision, recall))
            pr_auc = auc(recall, precision)
            aucs.append(pr_auc)

        if model == 'SVM':
            model = 'All'
        elif model == '5':
            model = 'MMMF Top 5'
        elif model == '10':
            model = 'MMMF Top 10'
        elif model == '20':
            model = 'MMMF Top 20'
        elif model == '30':
            model = 'MMMF Top 30'
        else:
            pass

        mean_precision = np.mean(prs, axis=0)
        mean_auc = auc(mean_recall, mean_precision)
        std_auc = np.std(aucs)

        ax.plot(mean_precision, mean_recall, label=r'' + model + ' (PR AUC = %0.3f$\pm$%0.3f)' % (mean_auc, std_auc), lw=4, alpha=.8)

        plt.xlim([-0.01, 1.01])
        plt.ylim([-0.01, 1.01])

        plt.xticks(fontsize=30)
        plt.yticks(fontsize=30)

    title = '\nClass 0: {}, Class 1: {}'.format(model_name_2, model_name_1)

    plt.title(title, fontsize=30)
    plt.legend(loc="lower left", prop={'size': 24})

    return ax