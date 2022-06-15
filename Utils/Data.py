import os
import pickle
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE

# Classification Dataset
class Dataset:
    def __init__(self, data_type, mp):
        # Initialization
        self.data_type = data_type
        self.mp = mp

        # Load Index
        with open(os.path.join('./Data', str(mp), self.data_type, 'index.pickle'), 'rb') as fr:
            self.index = pickle.load(fr)

        # Load Dataset
        self.modality1, self.modality2, self.modality3 = [None, None, None]

    # Each CV Load Dataset
    def __call__(self, cv):
        # Each CV
        train_index = self.index['cv' + str(cv + 1)]['train']
        validation_index = self.index['cv' + str(cv + 1)]['val']
        test_index = self.index['cv' + str(cv + 1)]['test']

        # Load Dataset
        self.load_data(cv)

        # Label
        y_train = self.modality1.iloc[train_index, 1].values
        y_val = self.modality1.iloc[validation_index, 1].values
        y_test = self.modality1.iloc[test_index, 1].values

        return [self.modality1.iloc[:, 2:].values, self.modality2.iloc[:, 2:].values,
                self.modality3.iloc[:, 2:].values], \
               [train_index, validation_index, test_index], \
               [y_train, y_val, y_test]

    # Load Dataset
    def load_data(self, cv):
        root_path = os.path.join('./Data', str(self.mp), self.data_type)

        # Load Dataset
        if self.data_type == 'ADNI':
            self.modality1 = pd.read_csv(os.path.join(root_path, 'cv'+str(cv+1)+'smri.csv'), index_col=0)
            self.modality2 = pd.read_csv(os.path.join(root_path, 'cv'+str(cv+1)+'pet.csv'), index_col=0)
            self.modality3 = pd.read_csv(os.path.join(root_path, 'cv'+str(cv+1)+'gene.csv'), index_col=0)

        elif self.data_type == 'ROSMAP':
            self.modality1 = pd.read_csv(os.path.join(root_path, 'cv'+str(cv+1)+'GE.csv'), index_col=0)
            self.modality2 = pd.read_csv(os.path.join(root_path, 'cv'+str(cv+1)+'ME.csv'), index_col=0)
            self.modality3 = pd.read_csv(os.path.join(root_path, 'cv'+str(cv+1)+'MI.csv'), index_col=0)

        elif self.data_type == 'BRCA':
            self.modality1 = pd.read_csv(os.path.join(root_path, 'cv' + str(cv + 1) + 'gene.csv'), index_col=0)
            self.modality2 = pd.read_csv(os.path.join(root_path, 'cv' + str(cv + 1) + 'prot.csv'), index_col=0)
            self.modality3 = pd.read_csv(os.path.join(root_path, 'cv' + str(cv + 1) + 'cnv.csv'), index_col=0)

        elif self.data_type == 'KIRC':
            self.modality1 = pd.read_csv(os.path.join(root_path, 'cv' + str(cv + 1) + 'GE.csv'), index_col=0)
            self.modality2 = pd.read_csv(os.path.join(root_path, 'cv' + str(cv + 1) + 'ME.csv'), index_col=0)
            self.modality3 = pd.read_csv(os.path.join(root_path, 'cv' + str(cv + 1) + 'MI.csv'), index_col=0)

        else:
            self.modality1 = pd.read_csv(os.path.join(root_path, 'cv'+str(cv+1)+'GE.csv'), index_col=0)
            self.modality2 = pd.read_csv(os.path.join(root_path, 'cv'+str(cv+1)+'MI.csv'), index_col=0)
            self.modality3 = pd.read_csv(os.path.join(root_path, 'cv'+str(cv+1)+'CpGs.csv'), index_col=0)

# Feature Selection Dataset
class FS_Dataset:
    def __init__(self, data_type, mp, top_n, feature_selection, random_state=0):
        # Initialization
        self.data_type = data_type
        self.mp = mp
        self.top_n = top_n
        self.feature_selection = feature_selection
        self.random_state = random_state

        # Load Index
        with open(os.path.join('./Data', str(mp), self.data_type, 'index.pickle'), 'rb') as fr:
            self.index = pickle.load(fr)

    # Each CV Load Dataset
    def __call__(self, cv):
        # Each CV
        train_index = self.index['cv' + str(cv + 1)]['train']
        validation_index = self.index['cv' + str(cv + 1)]['val']
        test_index = self.index['cv' + str(cv + 1)]['test']

        # Load Dataset
        modality1, modality2, modality3 = self.load_data(cv)

        # Train Dataset
        m1_x_train = modality1.iloc[train_index, 2:].values
        m2_x_train = modality2.iloc[train_index, 2:].values
        m3_x_train = modality3.iloc[train_index, 2:].values
        y_train = modality1.iloc[train_index, 1].values

        # Validation Dataset
        m1_x_val = modality1.iloc[validation_index, 2:].values
        m2_x_val = modality2.iloc[validation_index, 2:].values
        m3_x_val = modality3.iloc[validation_index, 2:].values
        y_val = modality1.iloc[validation_index, 1].values

        # Test Dataset
        m1_x_test = modality1.iloc[test_index, 2:].values
        m2_x_test = modality2.iloc[test_index, 2:].values
        m3_x_test = modality3.iloc[test_index, 2:].values
        y_test = modality1.iloc[test_index, 1].values

        # Select Using Modality
        x_train = np.concatenate((m1_x_train, m2_x_train, m3_x_train), axis=1)
        x_val = np.concatenate((m1_x_val, m2_x_val, m3_x_val), axis=1)
        x_test = np.concatenate((m1_x_test, m2_x_test, m3_x_test), axis=1)

        # Oversampling
        smote = SMOTE(random_state=0)
        x_train, y_train = smote.fit_resample(x_train, y_train)

        return [x_train, x_val, x_test], [y_train, y_val, y_test]

    # Load Dataset
    def load_data(self, cv):
        # Random Seed
        np.random.seed(self.random_state)

        # Load Dataset
        # MMMF Feature Selection
        data_root_path = os.path.join('./Result/Feature_Selection_Data/', str(self.top_n), str(self.mp), self.data_type)

        if self.data_type == 'ADNI':
            modality1 = pd.read_csv(os.path.join(data_root_path, 'cv' + str(cv + 1) + 'smri.csv'), index_col=0)
            modality2 = pd.read_csv(os.path.join(data_root_path, 'cv' + str(cv + 1) + 'pet.csv'), index_col=0)
            modality3 = pd.read_csv(os.path.join(data_root_path, 'cv' + str(cv + 1) + 'gene.csv'), index_col=0)

        elif self.data_type == 'ROSMAP':
            modality1 = pd.read_csv(os.path.join(data_root_path, 'cv' + str(cv + 1) + 'GE.csv'), index_col=0)
            modality2 = pd.read_csv(os.path.join(data_root_path, 'cv' + str(cv + 1) + 'ME.csv'), index_col=0)
            modality3 = pd.read_csv(os.path.join(data_root_path, 'cv' + str(cv + 1) + 'MI.csv'), index_col=0)

        elif self.data_type == 'BRCA':
            modality1 = pd.read_csv(os.path.join(data_root_path, 'cv' + str(cv + 1) + 'gene.csv'), index_col=0)
            modality2 = pd.read_csv(os.path.join(data_root_path, 'cv' + str(cv + 1) + 'prot.csv'), index_col=0)
            modality3 = pd.read_csv(os.path.join(data_root_path, 'cv' + str(cv + 1) + 'cnv.csv'), index_col=0)

        elif self.data_type == 'KIRC':
            modality1 = pd.read_csv(os.path.join(data_root_path, 'cv' + str(cv + 1) + 'GE.csv'), index_col=0)
            modality2 = pd.read_csv(os.path.join(data_root_path, 'cv' + str(cv + 1) + 'ME.csv'), index_col=0)
            modality3 = pd.read_csv(os.path.join(data_root_path, 'cv' + str(cv + 1) + 'MI.csv'), index_col=0)

        else:
            modality1 = pd.read_csv(os.path.join(data_root_path, 'cv' + str(cv + 1) + 'GE.csv'), index_col=0)
            modality2 = pd.read_csv(os.path.join(data_root_path, 'cv' + str(cv + 1) + 'MI.csv'), index_col=0)
            modality3 = pd.read_csv(os.path.join(data_root_path, 'cv' + str(cv + 1) + 'CpGs.csv'), index_col=0)

        if self.feature_selection != 'Random':
            pass

        # Random Selection
        else:
            original_data_root_path = os.path.join('./Data', str(self.mp), self.data_type)
            if self.data_type == 'ADNI':
                original_modality1 = pd.read_csv(os.path.join(original_data_root_path, 'cv' + str(cv + 1) + 'smri.csv'), index_col=0)
                original_modality2 = pd.read_csv(os.path.join(original_data_root_path, 'cv' + str(cv + 1) + 'pet.csv'), index_col=0)
                original_modality3 = pd.read_csv(os.path.join(original_data_root_path, 'cv' + str(cv + 1) + 'gene.csv'), index_col=0)
            elif self.data_type == 'ROSMAP':
                original_modality1 = pd.read_csv(os.path.join(original_data_root_path, 'cv' + str(cv + 1) + 'GE.csv'), index_col=0)
                original_modality2 = pd.read_csv(os.path.join(original_data_root_path, 'cv' + str(cv + 1) + 'ME.csv'), index_col=0)
                original_modality3 = pd.read_csv(os.path.join(original_data_root_path, 'cv' + str(cv + 1) + 'MI.csv'), index_col=0)
            elif self.data_type == 'BRCA':
                original_modality1 = pd.read_csv(os.path.join(original_data_root_path, 'cv' + str(cv + 1) + 'gene.csv'), index_col=0)
                original_modality2 = pd.read_csv(os.path.join(original_data_root_path, 'cv' + str(cv + 1) + 'prot.csv'), index_col=0)
                original_modality3 = pd.read_csv(os.path.join(original_data_root_path, 'cv' + str(cv + 1) + 'cnv.csv'), index_col=0)
            elif self.data_type == 'KIRC':
                original_modality1 = pd.read_csv(os.path.join(original_data_root_path, 'cv' + str(cv + 1) + 'GE.csv'), index_col=0)
                original_modality2 = pd.read_csv(os.path.join(original_data_root_path, 'cv' + str(cv + 1) + 'ME.csv'), index_col=0)
                original_modality3 = pd.read_csv(os.path.join(original_data_root_path, 'cv' + str(cv + 1) + 'MI.csv'), index_col=0)
            else:
                original_modality1 = pd.read_csv(os.path.join(original_data_root_path, 'cv' + str(cv + 1) + 'GE.csv'), index_col=0)
                original_modality2 = pd.read_csv(os.path.join(original_data_root_path, 'cv' + str(cv + 1) + 'MI.csv'), index_col=0)
                original_modality3 = pd.read_csv(os.path.join(original_data_root_path, 'cv' + str(cv + 1) + 'CpGs.csv'), index_col=0)

            # Load Dataset with Random Selection
            modality1 = original_modality1[['Subject', 'Label'] + list(
                np.random.choice(original_modality1.columns.to_list()[2:], size=modality1.shape[1] - 2, replace=False))]

            modality2 = original_modality2[['Subject', 'Label'] + list(
                np.random.choice(original_modality2.columns.to_list()[2:], size=modality2.shape[1] - 2, replace=False))]

            modality3 = original_modality3[['Subject', 'Label'] + list(
                np.random.choice(original_modality3.columns.to_list()[2:], size=modality3.shape[1] - 2, replace=False))]

        return modality1, modality2, modality3

# Pathway Dataset
class All_Dataset:
    def __init__(self, data_type):
        # Initialization
        self.data_type = data_type

        # Define Dataset
        self.modality1, self.modality2, self.modality3 = [None, None, None]

        # Load Dataset
        self.load_data()

    # Each CV Load Dataset
    def __call__(self):
        return [self.modality1.iloc[:, 2:].values, self.modality2.iloc[:, 2:].values,
                self.modality3.iloc[:, 2:].values], self.modality1.iloc[:, 1].values

    # Load Dataset
    def load_data(self):
        root_path = os.path.join('./Data/All/', self.data_type)

        # Load Dataset
        # ADNI
        if self.data_type == 'ADNI':
            self.modality1 = pd.read_csv(os.path.join(root_path, 'smri.csv'), index_col=0)
            self.modality2 = pd.read_csv(os.path.join(root_path, 'pet.csv'), index_col=0)
            self.modality3 = pd.read_csv(os.path.join(root_path, 'gene.csv'), index_col=0)
        # BRCA
        else:
            self.modality1 = pd.read_csv(os.path.join(root_path, 'gene.csv'), index_col=0)
            self.modality2 = pd.read_csv(os.path.join(root_path, 'prot.csv'), index_col=0)
            self.modality3 = pd.read_csv(os.path.join(root_path, 'cnv.csv'), index_col=0)