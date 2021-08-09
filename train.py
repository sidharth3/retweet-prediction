import numpy as np
import pandas as pd
import os
import sys
import shutil
from tqdm import tqdm

import mlflow
from sklearn.metrics import mean_squared_log_error
import torch
import torch.nn as nn
from tensorflow.python.keras.preprocessing.sequence import pad_sequences

from dataloader import Dataloader
from model import DNN
from model import MLP
from utils import label_inverse_scaling, label_scaling

def init_embeddings(feature_index, unique_num_dic, sparse_features, dense_features, varlen_sparse_features, device):
    VARLEN_MODE_LIST = ['mean']
    embedding_dict = nn.ModuleDict(
        {
            feat: nn.Embedding(
                unique_num_dic[feat], 40, sparse=False
            ) for feat in sparse_features
        }
    )
   
    for mode in VARLEN_MODE_LIST:
        for feat in varlen_sparse_features:
          
            embedding_dict[f'{feat}__{mode}'] = nn.Embedding(
                unique_num_dic[feat], 40, sparse=False
            )

    linear_embedding_dict = nn.ModuleDict(
        {
            feat: nn.Embedding(
                unique_num_dic[feat], 1, sparse=False
            ) for feat in sparse_features
        }
    )
    for mode in VARLEN_MODE_LIST:
        for feat in varlen_sparse_features:
            linear_embedding_dict[f'{feat}__{mode}'] = nn.Embedding(
                unique_num_dic[feat], 1, sparse=False
            )
    #print(embedding_dict)

    dnn_input_len = len(dense_features) + len(sparse_features) * 40 \
        + len(varlen_sparse_features) * 40

    model = MLP(
        dnn_input=dnn_input_len,
        dnn_hidden_units=(4096, 1028, 256),
        dnn_dropout=0.1,
        activation='relu', use_bn=True, l2_reg=1e-4, init_std=1e-4,
        device=device,
        feature_index=feature_index,
        embedding_dict=embedding_dict,
        dense_features=dense_features,
        sparse_features=sparse_features,
        varlen_sparse_features=varlen_sparse_features,
        varlen_mode_list=['mean'],
        embedding_size=40,
        batch_size=batch_size,
    )
    return model

def extract_saved_features(dense_features, sparse_features, varlen_sparse_features, FEATURE_DIR, VARLEN_MAX_LEN, y_train):


    scaler, y_train = label_scaling(y_train) ##import from utils

    unique_num_dic = {}
    feature_index = {}

    X_train = pd.DataFrame()
    X_valid = pd.DataFrame()
    X_test = pd.DataFrame()
    fidx = 0

    for feat in dense_features:
    
        feature_index[feat] = fidx
        fidx += 1
        X_train = pd.concat([X_train, pd.read_feather(f'{FEATURE_DIR}/{feat}_train.feather')], axis=1)
        
        X_valid = pd.concat([X_valid, pd.read_feather(f'{FEATURE_DIR}/{feat}_valid.feather')], axis=1)
        
        X_test = pd.concat([X_test, pd.read_feather(f'{FEATURE_DIR}/{feat}_test.feather')], axis=1)
    
    for feat in sparse_features:
        print(feat)  
        feature_index[feat] = fidx
        fidx += 1
        
        X_train = pd.concat([X_train, pd.read_feather(f'{FEATURE_DIR}/{feat}_train.feather')], axis=1)
        
        X_valid = pd.concat([X_valid, pd.read_feather(f'{FEATURE_DIR}/{feat}_valid.feather')], axis=1)
        
        X_test = pd.concat([X_test, pd.read_feather(f'{FEATURE_DIR}/{feat}_test.feather')], axis=1)
        unique_num = pd.concat([X_train[feat], X_valid[feat], X_test[feat]]).nunique()
        unique_num_dic[feat] = unique_num
    
    for feat in varlen_sparse_features:
        feature_index[feat] = (fidx, fidx + VARLEN_MAX_LEN)
        fidx += VARLEN_MAX_LEN

        train_feat = pd.read_feather(f'{FEATURE_DIR}/{feat}_train.feather').values
        varlen_list = [i[0] for i in train_feat]
        varlen_list = pad_sequences(varlen_list, maxlen=VARLEN_MAX_LEN, padding='post', )
        X_train = pd.concat([X_train, pd.DataFrame(varlen_list)], axis=1)

        valid_feat = pd.read_feather(f'{FEATURE_DIR}/{feat}_valid.feather').values
        varlen_list = [i[0] for i in valid_feat]
        varlen_list = pad_sequences(varlen_list, maxlen=VARLEN_MAX_LEN, padding='post', )
        X_valid = pd.concat([X_valid, pd.DataFrame(varlen_list)], axis=1)

        test_feat = pd.read_feather(f'{FEATURE_DIR}/{feat}_test.feather').values
        varlen_list = [i[0] for i in test_feat]
        varlen_list = pad_sequences(varlen_list, maxlen=VARLEN_MAX_LEN, padding='post', )
        X_test = pd.concat([X_test, pd.DataFrame(varlen_list)], axis=1)

        tmp = []
        for i in [i[0] for i in train_feat] + [i[0] for i in valid_feat] + [i[0] for i in test_feat]:
            tmp.extend(i)
        unique_num = len(set(tmp))
        unique_num_dic[feat] = unique_num
        
        unique_num = len(set(tmp))
        unique_num_dic[feat] = unique_num

    X_train = X_train.fillna(0.0)
    X_valid = X_valid.fillna(0.0)
    X_test = X_test.fillna(0.0)

    return X_train, X_valid, X_test, unique_num_dic, feature_index, scaler, y_train

def train(FOLD_DIR, SAVE_DIR, X_train, y_train, feature_index, unique_num_dic, scaler, batch_size, learning_rate, epochs, device, label_log_scaling,\
    sparse_features, dense_features, varlen_sparse_features):
    folds = pd.read_csv(f'{FOLD_DIR}/train_folds_1month_5fold10_RS45.csv', nrows=100000)
    FOLD_NUM = 5

    mlflow.set_experiment("EXP_NAME")
    mlflow.start_run()
    run_id = mlflow.active_run().info.run_id



    fold_best_scores = {} 
    for fold_idx in range(FOLD_NUM):

        trn_idx = folds[folds.kfold != fold_idx].index.tolist()
        val_idx = folds[folds.kfold == fold_idx].index.tolist()

        x_trn = X_train.iloc[trn_idx]
        y_trn = y_train[trn_idx]
        x_val = X_train.iloc[val_idx]
        y_val = y_train[val_idx]

        train_loader = Dataloader([torch.from_numpy(x_trn.values), torch.from_numpy(y_trn)], batch_size=batch_size,
                    shuffle=True)

        model = init_embeddings(feature_index=feature_index,unique_num_dic=unique_num_dic, sparse_features=sparse_features, dense_features=dense_features, varlen_sparse_features=varlen_sparse_features, device=torch.device('cpu'))
        
        loss_func = nn.MSELoss(reduction='mean')
        optim = torch.optim.Adam(model.parameters(), learning_rate=learning_rate)
        
        scheduler = torch.optim.learning_rate_scheduler.CosineAnnealinglearning_rate(optim, T_max=epochs)

        loss_history = []

        steps_per_epoch = (len(x_trn) - 1) // batch_size + 1
        best_score = 999.9

        for epoch in range(epochs):
            
            loss_history_epoch = []
            metric_history_epoch = []


            model = model.train()
            for bi, (bx, by) in tqdm(enumerate(train_loader), total=steps_per_epoch):

                optim.zero_grad()

                bx = bx.to(device).float()
                by = by.to(device).float().squeeze()
                y_pred = model(bx).squeeze()

                loss = 0.0
                for loss_f in loss_func:
                    loss += loss_f(y_pred, by)
                loss = loss + model.reg_loss.item()

                loss.backward()
                optim.step()

                y_pred_np = y_pred.cpu().detach().numpy().reshape(-1, 1)
                y_np = by.cpu().detach().numpy().reshape(-1, 1)

                try:
                    if label_log_scaling is True:
                        y_pred_inv = label_inverse_scaling(scaler, y_pred_np)
                        y_inv = label_inverse_scaling(scaler, y_np)
                        mlse = mean_squared_log_error(y_inv, y_pred_inv)
                    else:
                        mlse = mean_squared_log_error(y_np, y_pred_np)
                    loss_history_epoch.append(loss.item())
                    metric_history_epoch.append(mlse)
                except:
                    continue

            scheduler.step()
            trn_loss_epoch = sum(loss_history_epoch) / len(loss_history_epoch)
            trn_metric_epoch = sum(metric_history_epoch) / len(metric_history_epoch)

            preds_val = model.predict(x_val, batch_size)
            val_loss = 0.0
            for loss_f in loss_func:
                val_loss += loss_f(torch.from_numpy(preds_val.reshape(-1, 1)), torch.from_numpy(y_val)).item()

            try:
                if label_log_scaling is True:
                    preds_val_inv = label_inverse_scaling(scaler, preds_val.reshape(-1, 1))
                    y_val_inv = label_inverse_scaling(scaler, y_val)
                    val_metric = mean_squared_log_error(y_val_inv, preds_val_inv)
                else:
                    val_metric = mean_squared_log_error(y_val, preds_val)
            except:
                continue

        
            loss_history.append([
                epoch, trn_loss_epoch, trn_metric_epoch, val_loss, val_metric
            ])

            if val_metric < best_score:
                best_score = val_metric
                weight_path = f'{SAVE_DIR}/model/train_weights_mlflow-{run_id}_fold{fold_idx}.h5'
                torch.save(model.state_dict(), weight_path)
                fold_best_scores[fold_idx] = (best_score, weight_path)
                
        history_path = f'{SAVE_DIR}/model/loss_history-{run_id}_fold{fold_idx}.csv'
        pd.DataFrame(loss_history, columns=['epoch', 'trn_loss', 'trn_metric', 'val_loss', 'val_metric']).to_csv(history_path)
        mlflow.log_artifact(history_path)

def predict():
    pass



if __name__ == '__main__':

    INPUT_DIR = '/content/drive/MyDrive/dataset/data/'
    FEATURE_DIR = '/content/drive/MyDrive/dataset/feat_new/'
    FOLD_DIR = '/content/drive/MyDrive/dataset/fold/'
    SAVE_DIR = '/content/drive/MyDrive/dataset/save'
    SUB_DIR = '/content/drive/MyDrive/dataset/sub/'

    FOLD_NAME = '1month_5fold'
    FOLD_NUM = 5
    FOLD_IDX = 0
    RANDOM_STATE = 46

    MODEL_NAME = 'MLP'

    epochs = 10
    batch_size = 512
    DNN_HIDDEN_UNITS = (4096, 1024, 128)  # (2048, 512, 128)  # (4096, 1024, 128)
    DNN_DROPOUT = 0.1
    DNN_ACTIVATION = 'relu'
    L2_REG = 1e-4
    INIT_STD = 1e-4

    SPARSE_EMBEDDING_DIM = 40
    SPARSE_EMBEDDING = False
    VARLEN_MAX_LEN = 5
    VARLEN_MODE_LIST = ['mean']
    learning_rate = 0.001
    OPTIMIZER = 'adam'
    LOSS = 'MAE'  # 'MSE', 'MAE'

    label_log_scaling = True



    

