import numpy as np
import random
import pandas as pd
import matplotlib.pyplot as plt
import os
import copy
import seaborn as sn
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from dae_dataset import MoASwapDataset,TestDataset
from dae_models import Model

def seed_everything(seed=42):
    random.seed(seed)
    #os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def train_fn(model, optimizer, scheduler, loss_fn, dataloader, device):
    model.train()
    final_loss = 0
    
    for data in dataloader:
        optimizer.zero_grad()
        inputs, targets = data['x'].to(device), data['y'].to(device)
#         print(inputs.shape)
        outputs = model(inputs)
        loss = loss_fn(outputs, targets)
        loss.backward()
        optimizer.step()
        
        if not  scheduler.__class__ ==  torch.optim.lr_scheduler.ReduceLROnPlateau:
            scheduler.step()
        
        final_loss += loss.item()
        
    final_loss /= len(dataloader)
    
    return final_loss


def valid_fn(model, scheduler, loss_fn, dataloader, device):
    model.eval()
    final_loss = 0
    valid_preds = []
    
    for data in dataloader:
        inputs, targets = data['x'].to(device), data['y'].to(device)
        outputs = model(inputs)
        loss = loss_fn(outputs, targets)
        
        final_loss += loss.item()
        valid_preds.append(outputs.sigmoid().detach().cpu().numpy())
        
    final_loss /= len(dataloader)
    valid_preds = np.concatenate(valid_preds)
    
    if scheduler.__class__ ==  torch.optim.lr_scheduler.ReduceLROnPlateau:
        scheduler.step(final_loss)
    
    return final_loss, valid_preds

def inference_fn(model, dataloader, device):
    model.eval()
    preds = []
    
    for data in dataloader:
        inputs = data['x'].to(device)

        with torch.no_grad():
            outputs = model(inputs)
        
        preds.append(outputs.sigmoid().detach().cpu().numpy())
        
    preds = np.concatenate(preds)
    
    return preds
   

def run_k_fold(folds,target_cols,test,NFOLDS, seed,verbose=False,**kwargs):
    
    
    train = folds
    test_ = test
    
    #oof = np.zeros((len(folds), len(target_cols)))
    oof = train[target_cols].copy()
    predictions = np.zeros((len(test), len(target_cols)))
    
    #print(test_.head())
    for fold in range(NFOLDS):
        
        #trn_idx = train[train['kfold'] != fold].reset_index().index
        #val_idx = train[train['kfold'] == fold].reset_index().index
    
        train_df = train[train['fold_id'] != fold]#.reset_index(drop=True)
        valid_df = train[train['fold_id'] == fold]#.reset_index(drop=True)
        
       # print(len(train_df))
        #print(len(valid_df))
        
#         feature_cols = [col  for col in train_df.columns if not (col in target_cols+['kfold'])]
        feature_cols = target_cols
        #print(feature_cols)
        
        
        X_train, y_train  = train_df[feature_cols], train_df[target_cols]
        X_valid, y_valid =  valid_df[feature_cols], valid_df[target_cols].values
        
      
       
        feature_cols = [col  for col in X_train.columns if not (col in target_cols+['fold_id'])]
        
        X_train = X_train.values
        
        
        valid_index = X_valid.index
        X_valid = X_valid.values
        
        y_train = y_train.values
        
        # test set is useless here
        X_test = test_[feature_cols].values
            
        #oof_, pred_ = 
        run_training(X_train,X_valid,X_test,fold, seed,verbose,**kwargs)
        
        #oof.loc[valid_index] = oof_
        
        
        
        #predictions += pred_ / NFOLDS
        
        
    return #oof, predictions




def run_training(X_train,X_valid,X_test,fold, optim_dict,schedule_dict, device, seed,verbose=False,**kwargs):
    
    seed_everything(seed)
    
   
    
    train_dataset = MoASwapDataset(X_train, 0.2)
    valid_dataset = MoASwapDataset(X_valid, 0.2)
    trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    validloader = torch.utils.data.DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    model = Model(
        num_features= X_train.shape[1] ,
        num_targets=  X_train.shape[1],
        hidden_size=hidden_size,hidden_size2=hidden_size2,**kwargs
    )
    
    model.to(device)
    
    #initialize_from_past_model(model,f"../results/FOLD{fold}_original_torch_moa_5_folds_AUX.pth")#,freeze_first_layer=True)
    #optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    
    learning_rate = optim_dict['lr']
    wd =  optim_dict['weight_decay']
    
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=wd)
    #scheduler = optim.lr_scheduler.OneCycleLR(optimizer=optimizer, pct_start=0.1, div_factor=1e2, 
                                          #max_lr=5e-4, epochs=EPOCHS, steps_per_epoch=len(trainloader))
    
    
    fac = schedule_dict['factor']
    pat = schedule_dict['patience']
    thresh = schedule_dict['threshold']
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,patience=pat,factor=fac,threshold=thresh)
    #scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,patience=10,factor=0.5,threshold=1e2)
    #scheduler = optim.lr_scheduler.CyclicLR(optimizer,1e-4,5e-3,scale_mode='exp_range',gamma=FACTOR)
   # loss_val = nn.BCEWithLogitsLoss()
    
    #loss_tr = SmoothBCEwLogits(smoothing =0.001)
    
    
    loss_tr = nn.MSELoss()
    loss_val = nn.MSELoss()
    
    early_stopping_steps = EARLY_STOPPING_STEPS
    early_step = 0
    
   
    best_loss = np.inf
    
    
    
    
    for epoch in range(EPOCHS):
        
        train_loss = train_fn(model, optimizer,scheduler, loss_tr, trainloader, device)
        if verbose:
            if epoch %5 ==0: print(f"FOLD: {fold}, EPOCH: {epoch}, train_loss: {train_loss}")
        
        valid_loss, valid_preds = valid_fn(model,scheduler, loss_val, validloader, device)
        if verbose:
            if epoch %5 ==0: print(f"FOLD: {fold}, EPOCH: {epoch}, valid_loss: {valid_loss}")
        
        if valid_loss < best_loss:
            
            best_loss = valid_loss
            oof = valid_preds
        
        
            if epoch > 0.8*EPOCHS:
                torch.save(model.state_dict(), f"FOLD{fold}_{exp_name}.pth")
        
        elif(EARLY_STOP == True):
            
            early_step += 1
            if (early_step >= early_stopping_steps):
                break
            
    

    del model
    torch.cuda.empty_cache()
    return 

# generate latent features
def infer_features_fn(model, dataloader, device):
    model.eval()
    preds = []
    
    for data in dataloader:
        inputs = data['x'].to(device)

        with torch.no_grad():
            outputs = model(inputs,mode='get_features')
        

        preds.append(torch.cat(outputs,axis=1).detach().cpu().numpy())
        
        
        
    preds = np.concatenate(preds)
    
    return preds


def gen_latent_features(X_infer, fold, exp_name, model_arch_dict, b_size, dae_model_path, device, seed,inference_only=False,**kwargs):
    
    seed_everything(seed)
   
    infer_dataset = TestDataset(X_infer)
    infer_loader = torch.utils.data.DataLoader(infer_dataset, batch_size=b_size, shuffle=False)
    
    hidden_size1 = model_arch_dict[0]
    hidden_size2 = model_arch_dict[1]
    
    model = Model(
        num_features= X_infer.shape[1] ,
        num_targets=  X_infer.shape[1],
        hidden_size=hidden_size1,hidden_size2=hidden_size2,**kwargs
    )
    
    
    model.load_state_dict(torch.load( f"{dae_model_path}/FOLD{fold}_{exp_name}.pth",map_location=torch.device('cpu')))
    
    model.to(device)
    
    predictions = infer_features_fn(model, infer_loader , device)
    
    
    return predictions