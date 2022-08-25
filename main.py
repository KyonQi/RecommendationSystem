# if you don't have annoy librabry, please pip install it
# !pip install 
# Also, if you don't have torch-rechub library, please use following commands
# 1. git clone https://github.com/datawhalechina/torch-rechub.git
# 2. cd torch-rechub
# 3. python setup.py install

import annoy
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import sys
from utils import GenerateWholeCSV, match_evaluation
from model import YoutubeDNN, YDNNA
from sklearn.preprocessing import LabelEncoder
from torch_rechub.trainers import MatchTrainer
from torch_rechub.basic.features import SparseFeature, SequenceFeature
from torch_rechub.utils.match import generate_seq_feature_match, gen_model_input
from torch_rechub.utils.data import df_to_dict, MatchDataGenerator


if __name__ == '__main__':
    whole_data_path_local = './content/ml-1m/original' # please replce it with your own direction
    GenerateWholeCSV(whole_data_path_local)
    data_path = '/content/ml-1m/ml-1m.csv' # please replace it with your own direction
    data = pd.read_csv(data_path)
    #print(data.head())
    save_dir = '/content/ml-1m/rawdata/'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    data['cate_id'] = data['genres'].apply(lambda x: x.split('|')[0])
    sparse_features = ['user_id', 'movie_id', 'gender', 'age', 'occupation', 'zip', 'cate_id']
    user_col, item_col = 'user_id', 'movie_id'
    feature_max_idx = {} # it's a dict, describing the number of each feature
    for feature in sparse_features:
        lbe = LabelEncoder()
        data[feature] = lbe.fit_transform(data[feature]) + 1 # starting from 1 instead of 0
        feature_max_idx[feature] = data[feature].max() + 1
    if feature == user_col:
        user_map = {encode_id + 1 : raw_id for encode_id, raw_id in enumerate(lbe.classes_)}
    if feature == item_col:
        item_map = {encode_id + 1 : raw_id for encode_id, raw_id in enumerate(lbe.classes_)}

    np.save(save_dir + 'raw_id_maps.npy', (user_map, item_map))

    user_cols = ['gender', 'age', 'occupation', 'zip'] #not using user_id as user_cols
    item_cols = ['movie_id', 'cate_id']
    user_profile = data[ [user_col] + user_cols].drop_duplicates(user_col) #contains all the information of users
    item_profile = data[item_cols].drop_duplicates(item_col) #contains all the information of items
    #print(user_profile.head())
    #print(item_profile.head())

    df_train, df_test = generate_seq_feature_match(data, user_col, item_col, time_col='timestamp',
                         item_attribute_cols=[], sample_method=1, mode=2, 
                         neg_ratio=5, min_item=0) #generate the sequence feature based on timestamp
    #print(df_train.head(),'\n',len(df_train))
    x_train = gen_model_input(df_train, user_profile, user_col, item_profile, item_col, seq_max_len=50)
    del x_train[user_col] #as prof has said, user_id is not necessary for the input data
    y_train = np.array( [0] * df_train.shape[0] ) #[pos_sample, neg_sample, neg_sample, ..., neg_sample], so the label=0  it's a multi-task classification problem
    x_test = gen_model_input(df_test, user_profile, user_col, item_profile, item_col, seq_max_len=50)

    #Then we'll construct our YDNN and YDNNA model
    user_sparse_features = [SparseFeature(name, vocab_size=feature_max_idx[name], embed_dim=32) for name in user_cols]
    user_sequence_features = [SequenceFeature('hist_movie_id', vocab_size=feature_max_idx['movie_id'],
                      embed_dim=32, pooling='concat', shared_with='movie_id')]
    user_features = user_sparse_features + user_sequence_features

    #item_sparse_features = [SparseFeature(name, vocab_size=feature_max_idx[name], embed_dim=32) for name in item_cols] data could add a column with sequence of label
    item_sparse_features = [SparseFeature('movie_id', vocab_size=feature_max_idx['movie_id'], embed_dim=32)]
    item_sequence_features = []
    item_features = item_sparse_features + item_sequence_features

    neg_item_features = [SequenceFeature('neg_items', vocab_size=feature_max_idx['movie_id'], embed_dim=32,
                    pooling='concat', shared_with='movie_id')]
    
    all_item = df_to_dict(item_profile)
    test_user = x_test

    dg = MatchDataGenerator(x=x_train, y=y_train) #This dataloader class can be optimized better; it's not understandable and convenient enough
    train_dl, test_dl, item_dl = dg.generate_dataloader(test_user, all_item, batch_size=256, num_workers=2)

    model = YDNNA(user_features, item_features, neg_item_features, user_params={"dims": [128, 256, 128, 64, 32]}, attention_mlp_params={"dims": [256, 128]}, temperature=0.02)
    #model = YoutubeDNN(user_features, item_features, neg_item_features, user_params={"dims": [128, 256, 128, 64, 32]}, temperature=0.02)

    trainer = MatchTrainer(model, mode=2, optimizer_params={'lr': 1e-4, 'weight_decay': 1e-6},
             n_epoch=10, device='cuda:0', model_path=save_dir)
    trainer.fit(train_dl)

    print('inference embedding')
    user_embedding = trainer.inference_embedding(model=model, mode='user', data_loader=test_dl, model_path=save_dir)
    item_embedding = trainer.inference_embedding(model=model, mode='item', data_loader=item_dl, model_path=save_dir)
    metric_dict, y_pred, y_truth = match_evaluation(user_embedding, item_embedding, test_user, all_item, topk=10, raw_id_maps=save_dir+"raw_id_maps.npy")