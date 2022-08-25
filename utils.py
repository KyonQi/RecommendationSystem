import collections
import pandas as pd
import numpy as np
from torch_rechub.utils.match import Annoy
from torch_rechub.basic.metric import topk_metrics
from collections import Counter

# First, please get MovieLens-1M dataset in your own computer and change the following direction
def GenerateWholeCSV(original_data_dir):
    '''This function is used to generate the CSV file of MovieLens-1M dataset.
    Args: 
        original_data_dir (File direction): The absolute direction of original MovieLens-1M file
    '''
    data_path = original_data_dir

    unames = ['user_id','gender','age','occupation','zip']
    user = pd.read_csv(data_path+'/ml-1m/users.dat',sep='::',header=None,names=unames, engine='python',encoding="ISO-8859-1")
    rnames = ['user_id','movie_id','rating','timestamp']
    ratings = pd.read_csv(data_path+'/ml-1m/ratings.dat',sep='::',header=None,names=rnames, engine='python',encoding="ISO-8859-1")
    mnames = ['movie_id','title','genres']
    movies = pd.read_csv(data_path+'/ml-1m/movies.dat',sep='::',header=None,names=mnames, engine='python',encoding="ISO-8859-1")

    data = pd.merge(pd.merge(ratings,movies),user)
    data.to_csv("../content/ml-1m/ml-1m.csv", index=False)

def match_evaluation(user_embedding, item_embedding, test_user, all_item, user_col='user_id', item_col='movie_id',
                     raw_id_maps="./data/ml-1m/saved/raw_id_maps.npy", topk=10):
    print("evaluate embedding matching on test data")
    annoy = Annoy(n_trees=10)
    annoy.fit(item_embedding)

    #for each user of test dataset, get ann search topk result
    print("matching for topk")
    user_map, item_map = np.load(raw_id_maps, allow_pickle=True)
    match_res = collections.defaultdict(dict)  # user id -> predicted item ids
    for user_id, user_emb in zip(test_user[user_col], user_embedding):
        if len(user_emb.shape)==2:
            #multi-recall
            items_idx = []
            items_scores = []
            for i in range(user_emb.shape[0]):
                temp_items_idx, temp_items_scores = annoy.query(v=user_emb[i], n=topk)  # the index of topk match items
                items_idx += temp_items_idx
                items_scores += temp_items_scores
            temp_df = pd.DataFrame()
            temp_df['item'] = items_idx
            temp_df['score'] = items_scores
            temp_df = temp_df.sort_values(by='score', ascending=True)
            temp_df = temp_df.drop_duplicates(subset=['item'], keep='first', inplace=False)
            recall_item_list = temp_df['item'][:topk].values
            match_res[user_map[user_id]] = np.vectorize(item_map.get)(all_item[item_col][recall_item_list])
        else:
            #single-recall
            items_idx, items_scores = annoy.query(v=user_emb, n=topk)  #the index of topk match items
            match_res[user_map[user_id]] = np.vectorize(item_map.get)(all_item[item_col][items_idx])

    #get ground truth
    print("generate ground truth")

    data = pd.DataFrame({user_col: test_user[user_col], item_col: test_user[item_col]})
    data[user_col] = data[user_col].map(user_map)
    data[item_col] = data[item_col].map(item_map)
    user_pos_item = data.groupby(user_col).agg(list).reset_index()
    ground_truth = dict(zip(user_pos_item[user_col], user_pos_item[item_col]))  # user id -> ground truth

    print("compute topk metrics")
    out = topk_metrics(y_true=ground_truth, y_pred=match_res, topKs=[topk])
    print(out)
    return out, match_res, ground_truth

if __name__ == '__main__':
    pd.set_option('display.max_rows', 500)
    pd.set_option('display.max_columns', 500)
    pd.set_option('display.width', 1000)
    whole_data_path_local = './content/ml-1m/original' # please replce it with your own direction
    GenerateWholeCSV(whole_data_path_local)
    data_path = '/content/ml-1m/ml-1m.csv'
    data = pd.read_csv(data_path)
    print(data.head())
