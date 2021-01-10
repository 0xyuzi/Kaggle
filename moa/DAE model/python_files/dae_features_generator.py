import sys
# sys.path.append('../input/iterative-stratification/iterative-stratification-master')
#sys.path.append('../..')
# from iterstrat.ml_stratifiers import MultilabelStratifiedKFold



import numpy as np
import random
import pandas as pd
import matplotlib.pyplot as plt
import os
import copy
import seaborn as sn


from sklearn import preprocessing
from sklearn.metrics import log_loss
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.preprocessing import QuantileTransformer
from sklearn.pipeline import make_pipeline,make_union
from sklearn.feature_selection import VarianceThreshold

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


from dae_utils import gen_latent_features

import warnings
warnings.filterwarnings('ignore')

work_dir = './'


def dae_features_generator(dae_model_path='../input/zip-model-1/dae_model_params'):
    input_directory = '../input/lish-moa/'

    # load data
    print("load datasets")
    train_features = pd.read_csv('../input/lish-moa/train_features.csv')
    train_targets_scored = pd.read_csv('../input/lish-moa/train_targets_scored.csv')
    train_targets_nonscored = pd.read_csv('../input/lish-moa/train_targets_nonscored.csv')
    test_features = pd.read_csv('../input/lish-moa/test_features.csv')


    GENES = [col for col in train_features.columns if col.startswith('g-')]
    CELLS = [col for col in train_features.columns if col.startswith('c-')]

    from sklearn.decomposition import PCA
    from sklearn.preprocessing import QuantileTransformer

    def process_data(data):
        data = pd.get_dummies(data, columns=['cp_time','cp_dose'])
        return data
    
    def seed_everything(seed=42):
        random.seed(seed)
        #os.environ['PYTHONHASHSEED'] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True

    seed_everything(seed=42)


    # PCA parameters
    n_comp_GENES = 463
    n_comp_CELLS = 60
    VarianceThreshold_for_FS = 0.01


    # RankGauss - transform to Gauss
    print("start rankguass")
    for col in (GENES + CELLS):

        transformer = QuantileTransformer(n_quantiles=100,random_state=0, output_distribution="normal")
        vec_len = len(train_features[col].values)
        vec_len_test = len(test_features[col].values)
        raw_vec = train_features[col].values.reshape(vec_len, 1)
        transformer.fit(raw_vec)

        train_features[col] = transformer.transform(raw_vec).reshape(1, vec_len)[0]
        test_features[col] = transformer.transform(test_features[col].values.reshape(vec_len_test, 1)).reshape(1, vec_len_test)[0]

    print("complete rankguass")

    # GENES
    print("start PCA for genes and cells")
    data = pd.concat([pd.DataFrame(train_features[GENES]), pd.DataFrame(test_features[GENES])])
    data2 = (PCA(n_components=n_comp_GENES, random_state=42).fit_transform(data[GENES]))
    train2 = data2[:train_features.shape[0]]; test2 = data2[-test_features.shape[0]:]

    train2 = pd.DataFrame(train2, columns=[f'pca_G-{i}' for i in range(n_comp_GENES)])
    test2 = pd.DataFrame(test2, columns=[f'pca_G-{i}' for i in range(n_comp_GENES)])

    train_features = pd.concat((train_features, train2), axis=1)
    test_features = pd.concat((test_features, test2), axis=1)


    # CELLS

    data = pd.concat([pd.DataFrame(train_features[CELLS]), pd.DataFrame(test_features[CELLS])])
    data2 = (PCA(n_components=n_comp_CELLS, random_state=42).fit_transform(data[CELLS]))
    train2 = data2[:train_features.shape[0]]; test2 = data2[-test_features.shape[0]:]

    train2 = pd.DataFrame(train2, columns=[f'pca_C-{i}' for i in range(n_comp_CELLS)])
    test2 = pd.DataFrame(test2, columns=[f'pca_C-{i}' for i in range(n_comp_CELLS)])

    train_features = pd.concat((train_features, train2), axis=1)
    test_features = pd.concat((test_features, test2), axis=1)


    var_thresh = VarianceThreshold(VarianceThreshold_for_FS)
    data = train_features.append(test_features)
    data_transformed = var_thresh.fit_transform(data.iloc[:, 4:])

    train_features_transformed = data_transformed[ : train_features.shape[0]]
    test_features_transformed = data_transformed[-test_features.shape[0] : ]


    train_features = pd.DataFrame(train_features[['sig_id','cp_type','cp_time','cp_dose']].values.reshape(-1, 4),\
                                  columns=['sig_id','cp_type','cp_time','cp_dose'])

    train_features = pd.concat([train_features, pd.DataFrame(train_features_transformed)], axis=1)


    test_features = pd.DataFrame(test_features[['sig_id','cp_type','cp_time','cp_dose']].values.reshape(-1, 4),\
                                 columns=['sig_id','cp_type','cp_time','cp_dose'])

    test_features = pd.concat([test_features, pd.DataFrame(test_features_transformed)], axis=1)

    print("complete PCA for genes and cells")


    # train = train_features.merge(train_targets_scored, on='sig_id')
    train = train_features
    train = train[train['cp_type']!='ctl_vehicle'].reset_index(drop=True)
    test = test_features[test_features['cp_type']!='ctl_vehicle'].reset_index(drop=True)


    train = train.drop('cp_type', axis=1)
    test = test.drop('cp_type', axis=1)


    # load the drug_id CV fold
    print("load the drug_id CV fold")
    drug_id_df = cv_drug_df = pd.read_csv("../input/cv-fold-data/cv-10fold-with-drug_id.csv")







    target_cols = [c for c in process_data(train).columns]
    target_cols = [c for c in target_cols if c not in ['fold_id','sig_id']]

    
    train_cv_merge= train.merge(drug_id_df, on='sig_id')
    train_cv_merge = train_cv_merge.drop(columns = ['drug_id'])

    train_cv_merge = process_data(train_cv_merge)
    test = process_data(test) 
    test_cv = test.copy()
    test_cv['fold_id'] = 10


    dae_merge_df = pd.concat([train_cv_merge, test_cv])


    # hyperparameters

    DEVICE = ('cuda' if torch.cuda.is_available() else 'cpu')
    EPOCHS = 300
    BATCH_SIZE = 512
    LEARNING_RATE = 2e-4
    WEIGHT_DECAY = 2e-5
    NFOLDS = 11 # defined already because using drug_cv + test_cv for DAE training 
    EARLY_STOPPING_STEPS = 10
    EARLY_STOP = False
    GAMMA=0.5
    FACTOR=0.75
    #num_features=len(feature_cols)
    #num_targets=len(target_cols)
    hidden_size=1500
    hidden_size2=1500
    PATIENCE=10
    THRESHOLD = 5e-3

    model_arch_list = [hidden_size, hidden_size2]

    exp_name =  "test_DAE_all_together"



    SEED = [0]


    inference_features = np.zeros((len(dae_merge_df), 4500))
    transform_feature_list = []

    print("start generating DAE latent features")
    for seed in SEED:



           for fold in range(NFOLDS):



                # infer_fold_df = dae_merge_df[dae_merge_df['fold_id'] == fold]
                # infer all with NFLODS model params and avg them 
                infer_fold_df = dae_merge_df

                feature_cols = target_cols


                X_infer  = infer_fold_df[feature_cols]

                X_infer = X_infer.values


                # # gen_latent_features(X_infer, fold, exp_name, dae_model_path seed,inference_only=False,**kwargs)
                
                pred_ = gen_latent_features(X_infer,fold,exp_name,  model_arch_list, BATCH_SIZE,dae_model_path, DEVICE, seed,inference_only=True)
#                 print(pred_.shape)
                transformed_features = pd.DataFrame(pred_,index=infer_fold_df.sig_id)
                transform_feature_list.append(transformed_features)

                # avg the latent vectors
                inference_features += pred_ /NFOLDS

    print("complete generating DAE latent features")
    inference_features_df = pd.DataFrame(inference_features,index=dae_merge_df.sig_id)
    dae_merge_sig_fold_df =  dae_merge_df[['sig_id', 'fold_id']]
    total_latent_features = inference_features_df.merge(dae_merge_sig_fold_df, on ='sig_id')
    total_latent_features.columns = [str(i) for i in total_latent_features.columns]
    # total_latent_features.to_feather(f'{work_dir}features_trial_altogether.fth')
    print("complete outputing DAE latent features")
    return  total_latent_features

if __name__ == "__main__":
    dae_features_generator(dae_model_path='../input/zip-model-1/dae_model_params')