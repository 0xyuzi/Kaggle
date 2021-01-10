import sys
sys.path.append('../../input/iterative-stratification')

import numpy as np
import random
import pandas as pd
import os
import copy
import gc

import matplotlib.pyplot as plt 
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go

from sklearn import preprocessing
from sklearn.metrics import log_loss
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.preprocessing import QuantileTransformer
from sklearn.feature_selection import VarianceThreshold
# from iterstrat.ml_stratifiers import MultilabelStratifiedKFold

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.modules.loss import _WeightedLoss

import warnings
warnings.filterwarnings('ignore')

work_dir = './'

os.listdir('../input/lish-moa')

pd.set_option('max_columns', 2000)

from final_utils import predict_k_fold

from dae_features_generator import dae_features_generator
def final_prediction( model_path='../input/zip-model-1/final_model_params'):

    # train_features = pd.read_csv('../input/lish-moa/train_features.csv')
    train_targets_scored = pd.read_csv('../input/lish-moa/train_targets_scored.csv')
    # train_targets_nonscored = pd.read_csv('../input/lish-moa/train_targets_nonscored.csv')

    # test_features = pd.read_csv('../input/lish-moa/test_features.csv')
    sample_submission = pd.read_csv('../input/lish-moa/sample_submission.csv')


    def seed_everything(seed=42):
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True

    seed_everything(seed=42)



    NFOLDS = 10


    # load the DAE latent features

    total_latent_features = dae_features_generator(dae_model_path='../input/zip-model-1/dae_model_params')
    # DAE_features = pd.read_feather(dea_filename).set_index('sig_id')

    DAE_features = total_latent_features.set_index('sig_id')

    DAE_features = DAE_features.add_prefix('denoise_')
    DAE_features_cols = [c for c in DAE_features.columns if c not in ['denoise_fold_id']]
    
   
    feature_plus_dae_cols = DAE_features_cols



    train = DAE_features[DAE_features['denoise_fold_id']!=10]
    test = DAE_features[DAE_features['denoise_fold_id']==10]


    train =train.merge(train_targets_scored, on='sig_id')
    test = test.merge(sample_submission, on='sig_id')


    target_cols = train_targets_scored.drop('sig_id', axis=1).columns.values.tolist()

    target = train[train_targets_scored.columns]





    # HyperParameters

    DEVICE = ('cuda' if torch.cuda.is_available() else 'cpu')
    EPOCHS = 50
    BATCH_SIZE = 512
    LEARNING_RATE = 1e-5
    WEIGHT_DECAY = 2e-5
    # NFOLDS = 10 #7 defined above 
    EARLY_STOPPING_STEPS = 10
    EARLY_STOP = False

    num_features=len(DAE_features_cols )
    num_targets=len(target_cols)
    hidden_size=1000
    
    model_dict = dict()
    model_dict['num_features'] = num_features
    model_dict['num_targets'] = num_targets
    model_dict['hidden_size'] = hidden_size 

    # Averaging on multiple SEEDS

    SEED = [0, 1, 2, 3, 4] #5
    predictions = np.zeros((len(test), len(target_cols)))

    for seed in SEED:

        predictions_ = predict_k_fold(test,DAE_features_cols, target,target_cols, BATCH_SIZE, model_dict, model_path, NFOLDS, DEVICE, seed)
        predictions += predictions_ / len(SEED)

    test[target_cols] = predictions



    sub = sample_submission.drop(columns=target_cols).merge(test[['sig_id']+target_cols], on='sig_id', how='left').fillna(0)
    sub.to_csv(f'{work_dir}submission.csv', index=False)


if __name__ == "__main__":
    final_prediction(model_path='../input/zip-model-1/final_model_params')