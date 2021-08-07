import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import KFold
from xfeat import TargetEncoder, CountEncoder
from sklearn.cluster import KMeans

#feature extraction
def init_features():
    tweet_metrics_features = ['#followers', '#friends', '#favorites','#followers__#favorites', '#friends__#favorites', '#followers__#friends__#favorites']
    #log
    tweet_metrics_log_features = [f'{feat}_log' for feat in tweet_metrics_features]
    #rank
    tweet_metrics_rank_features = [f'{feat}_rank' for feat in tweet_metrics_features]

    time_cat_features = ['weekday', 'hour', 'day', 'week_of_month']
    time_num_features = ['diff_from_latest'] 

    to_encode_count = ['sentiment_pos', 'sentiment_neg','weekday', 'hour', 'day', 'week_of_month'] 

    count_encoded_features = [f'{feat}_ce' for feat in to_encode_count]

    to_encode_target = ['username', 'sentiment_pos', 'sentiment_neg', 'weekday', 'hour', 'day', 'week_of_month',
        '#followers_qbin_10', '#friends_qbin_10', '#favorites_qbin_10', '#followers__#favorites_qbin_10'] #qbin = quantile binning is #target_encording_source

    target_encoded_features = [f'{feat}_te' for feat in to_encode_target]

    #need to dimensionality reduction 
    # we are using SVD -> dim = 5
    n_gram = 1
    dim_red_name = "svd"
    dim_red_dim = 5

    text_tfidf_features = [f'TFIDF_{dim_red_name}_{i}' for i in range(dim_red_dim)]

    nbin = 10
    tweet_metrics_bin_features = [f'{feat}_qbin_{nbin}' for feat in tweet_metrics_features]

    varlen_ori_features = ['entities', 'mentions', 'hashtags', 'urls']

    varlen_count_features = [f'{feat}_count' for feat in varlen_ori_features]

    varlen_count_encoding_source = ['entities', 'mentions', 'hashtags', 'urls']

    varlen_count_encoding_features = [f'{feat}_ce' for feat in varlen_count_encoding_source]

    varlen_target_encoding_source = ['entities', 'mentions', 'hashtags', 'urls']

    varlen_target_encoding_features = [f'{feat}_te' for feat in varlen_target_encoding_source]

    user_features = ['username']

    sentiment_features = ['sentiment_pos','sentiment_neg']

    user_stats_features = [
        'inday_fol_increase', 'prevday_fol_increase', 'inweek_fol_increase', 'prevweek_fol_increase',
        'inday_fri_increase', 'prevday_fri_increase', 'inweek_fri_increase', 'prevweek_fri_increase',
        'user_follow_mean', 'user_follow_std',
        'user_friend_mean', 'user_friend_std',
        'user_favorite_mean', 'user_favorite_std',
        'user_entities_unique', 'user_mentions_unique', 'user_hashtags_unique', 'user_urls_unique'
    ]

    user_entites_features = [f'entites_TFIDF_{dim_red_name}_{i}' for i in range(dim_red_dim)]
    user_mentions_features = [f'mentions_TFIDF_{dim_red_name}_{i}' for i in range(dim_red_dim)]
    user_hashtags_features = [f'hashtags_TFIDF_{dim_red_name}_{i}' for i in range(dim_red_dim)]


    #need to do clustering
    USER_CLUSTER_NUM = 1000
    user_clustering_features = [f'user_stats_cluster_{USER_CLUSTER_NUM}', f'user_topic_cluster_{USER_CLUSTER_NUM}', f'user_stats_topic_cluster_{USER_CLUSTER_NUM}']
    #print(user_clustering_features)
    #DENSE FEATURES
    dense_features = []
    dense_features += tweet_metrics_features
    dense_features += tweet_metrics_log_features
    dense_features += tweet_metrics_rank_features
    dense_features += time_num_features
    dense_features += count_encoded_features
    dense_features += target_encoded_features
    dense_features += text_tfidf_features
    dense_features += varlen_count_encoding_features
    dense_features += varlen_target_encoding_features
    dense_features += user_stats_features

    #SPARSE FEATURES
    sparse_features = []
    sparse_features += user_features
    sparse_features += sentiment_features
    sparse_features += tweet_metrics_bin_features
    sparse_features += time_cat_features
    sparse_features += varlen_count_features
    sparse_features += user_clustering_features

    #variable length FEATURES
    varlen_sparse_features = []
    varlen_sparse_features += varlen_ori_features

    return dense_features, sparse_features, varlen_sparse_features, to_encode_count, to_encode_target,\
         user_stats_features, time_cat_features, time_num_features, tweet_metrics_features, tweet_metrics_log_features,\
              tweet_metrics_bin_features
             

def preprocess(feature_dir, X_train, X_valid, X_test, time_cat_features, time_num_features, \
    tweet_metrics_features, tweet_metrics_log_features, tweet_metrics_bin_features, to_encode_count, \
        y_train, to_encode_target, varlen_ori_features, user_stats_features, dense_features, sparse_features):
    
    X_train = extract_time_features(X_train)
    X_valid = extract_time_features(X_valid)
    X_test = extract_time_features(X_test)

    for feat in time_cat_features + time_num_features:
        save_as_feather(feat, feature_dir, X_train, X_valid, X_test)
    
    X_train[['#followers', '#friends', '#favorites']] = X_train[['#followers', '#friends', '#favorites']].astype(int)
    X_valid[['#followers', '#friends', '#favorites']] = X_valid[['#followers', '#friends', '#favorites']].astype(int)
    X_test[['#followers', '#friends', '#favorites']] = X_test[['#followers', '#friends', '#favorites']].astype(int)

    X_train['#followers__#favorites'] = X_train['#followers'] * X_train['#favorites']
    X_train['#friends__#favorites'] = X_train['#friends'] * X_train['#favorites']
    X_train['#followers__#friends__#favorites'] = X_train['#followers'] * X_train['#friends'] * X_train['#favorites']

    X_valid['#followers__#favorites'] = X_valid['#followers'] * X_valid['#favorites']
    X_valid['#friends__#favorites'] = X_valid['#friends'] * X_valid['#favorites']
    X_valid['#followers__#friends__#favorites'] = X_valid['#followers'] * X_valid['#friends'] * X_valid['#favorites']

    X_test['#followers__#favorites'] = X_test['#followers'] * X_test['#favorites']
    X_test['#friends__#favorites'] = X_test['#friends'] * X_test['#favorites']
    X_test['#followers__#friends__#favorites'] = X_test['#followers'] * X_test['#friends'] * X_test['#favorites']

    #LOG features -> apply log operation
    X_train[tweet_metrics_log_features] = (X_train[tweet_metrics_features] + 1).apply(np.log)
    X_valid[tweet_metrics_log_features] = (X_valid[tweet_metrics_features] + 1).apply(np.log)
    X_test[tweet_metrics_log_features] = (X_test[tweet_metrics_features] + 1).apply(np.log)

    X_all = pd.concat([X_train, X_valid, X_test])
    X_all[tweet_metrics_features] = X_all[tweet_metrics_features].astype(int)
    
    for feat in tweet_metrics_features:
        mean = X_all[feat].mean()
        std = X_all[feat].std()
        
        X_all[f'{feat}_rank'] = X_all[feat].rank(method='min')
        X_train[f'{feat}_rank'] = X_all.iloc[:len(X_train), :][f'{feat}_rank']
        X_valid[f'{feat}_rank'] = X_all.iloc[len(X_train):len(X_train) + len(X_valid), :][f'{feat}_rank']
        X_test[f'{feat}_rank'] = X_all.iloc[len(X_train) + len(X_valid):, :][f'{feat}_rank']
    
    for feat in tweet_metrics_features:
        X_train, X_valid, X_test = quantile_binning(feat, X_train, X_valid, X_test, nbin=nbin)

    for feat in tweet_metrics_bin_features:
        save_as_feather(feat, feature_dir, X_train, X_valid, X_test)

    X_train = mod_sentiment(X_train)
    X_valid = mod_sentiment(X_valid)
    X_test = mod_sentiment(X_test)

    X_train = preprocess_entities(X_train) #puts the entities as a list
    X_valid = preprocess_entities(X_valid)
    X_test = preprocess_entities(X_test)

    #PREPROCESS THE variable length features (hashtag, entities, etc)
    X_train['mentions'] = X_train['mentions'].astype(str)
    X_valid['mentions'] = X_valid['mentions'].astype(str)
    X_test['mentions'] = X_test['mentions'].astype(str)

    X_train = preprocess_varlen(X_train, 'mentions')
    X_valid = preprocess_varlen(X_valid, 'mentions')
    X_test = preprocess_varlen(X_test, 'mentions')

    X_train['hashtags'] = X_train['hashtags'].astype(str)
    X_valid['hashtags'] = X_valid['hashtags'].astype(str)
    X_test['hashtags'] = X_test['hashtags'].astype(str)

    X_train = preprocess_varlen(X_train, 'hashtags')
    X_valid = preprocess_varlen(X_valid, 'hashtags')
    X_test = preprocess_varlen(X_test, 'hashtags')

    X_train = preprocess_varlen(X_train, 'urls')
    X_valid = preprocess_varlen(X_valid, 'urls')
    X_test = preprocess_varlen(X_test, 'urls')

    for feat in varlen_ori_features:
        X_train[f'{feat}_count'] = X_train[feat].apply(varlen_count)
        X_valid[f'{feat}_count'] = X_valid[feat].apply(varlen_count)
        X_test[f'{feat}_count'] = X_test[feat].apply(varlen_count)
    
    X_train, X_valid, X_test = extract_text_tfidf(X_train, X_valid, X_test)
    
    encoder = CountEncoder(input_cols=to_encode_count)

    X_train = encoder.fit_transform(X_train)
    X_valid = encoder.transform(X_valid)
    X_test = encoder.transform(X_test)

    scaler, y_train = label_scaling(y_train)
    X_train['target'] = np.ravel(y_train)

    fold = KFold(n_splits=5, random_state=45, shuffle=True)

    encoder = TargetEncoder(input_cols=to_encode_target, target_col='target',fold=fold)
    
    X_train = encoder.fit_transform(X_train)
    X_valid = encoder.transform(X_valid)
    X_test = encoder.transform(X_test)

    for feat in varlen_ori_features:
        X_train, X_valid, X_test = varlen_count_encording(feat, X_train, X_valid, X_test)

    scaler, y_train = label_scaling(y_train)
    fold = KFold(n_splits=5, random_state=45, shuffle=True)


    for feat in varlen_ori_features:
        X_train, X_valid, X_test = varlen_target_encording(feat, X_train, X_valid, X_test, y_train, fold)


    import datetime
    X_all = pd.concat([X_train, X_valid, X_test], axis=0)

    timestamp = []

    for x in tqdm(X_all['timestamp'].values, total=len(X_all)):
        timestamp.append(datetime.datetime.strptime(' '.join(x.split(' ')[1:]), '%b %d %H:%M:%S %z %Y'))

    X_all['timestamp'] = timestamp

    inday_fol_increase_features = np.zeros(len(X_all))
    prevday_fol_increase_features = np.zeros(len(X_all))
    inweek_fol_increase_features = np.zeros(len(X_all))
    prevweek_fol_increase_features = np.zeros(len(X_all))

    inday_fri_increase_features = np.zeros(len(X_all))
    prevday_fri_increase_features = np.zeros(len(X_all))
    inweek_fri_increase_features = np.zeros(len(X_all))
    prevweek_fri_increase_features = np.zeros(len(X_all))

    user_follow_mean_features = np.zeros(len(X_all))
    user_follow_std_features = np.zeros(len(X_all))
    user_friend_mean_features = np.zeros(len(X_all))
    user_friend_std_features = np.zeros(len(X_all))
    user_favorite_mean_features = np.zeros(len(X_all))
    user_favorite_std_features = np.zeros(len(X_all))

    user_entities_unique_features = np.zeros(len(X_all))
    user_mentions_unique_features = np.zeros(len(X_all))
    user_hashtags_unique_features = np.zeros(len(X_all))
    user_urls_unique_features = np.zeros(len(X_all))
    user_num = X_all['username'].nunique()
    
    for idx, (username, X_user) in tqdm(enumerate(X_all.groupby('username')), total=user_num):

        X_user = X_user.sort_values('timestamp', ascending=True)
        user_index = X_user.index.tolist()

        date_fol_tmp = []
        week_fol_tmp = []
        date_fri_tmp = []
        week_fri_tmp = []
        follow_tmp = []
        friend_tmp = []
        fav_tmp = []
        entities_tmp, mentions_tmp, hashtags_tmp, urls_tmp = [], [], [], []
        date_tmp = ''
        week_tmp = ''

        rows = []
        for dt, fol, fri, fav, entities, mentions, hashtags, urls in X_user[['timestamp', '#followers', '#friends', '#favorites', 'entities', 'mentions', 'hashtags', 'urls']].values:

            if date_tmp != dt.date():
                date_tmp = dt.date()
                date_fol_tmp.append(fol)
                date_fri_tmp.append(fri)
            if week_tmp != dt.week:
                week_tmp = dt.week
                week_fol_tmp.append(fol)
                week_fri_tmp.append(fri)
            follow_tmp.append(fol)
            friend_tmp.append(fri)
            fav_tmp.append(fav)
            entities_tmp.extend(entities)
            mentions_tmp.extend(mentions)
            hashtags_tmp.extend(hashtags)
            urls_tmp.extend(urls)

            inday_fol_increase = fol - date_fol_tmp[-1]
            inweek_fol_increase = fol - week_fol_tmp[-1]
            inday_fri_increase = fri - date_fri_tmp[-1]
            inweek_fri_increase = fri - week_fri_tmp[-1]

            prevday_fol_increase = 0
            prevweek_fol_increase = 0
            if len(date_fol_tmp) > 1:
                prevday_fol_increase = date_fol_tmp[-1] - date_fol_tmp[-2]
            if len(week_fol_tmp) > 1:
                prevweek_fol_increase = week_fol_tmp[-1] - week_fol_tmp[-2]

            prevday_fri_increase = 0
            prevweek_fri_increase = 0
            if len(date_fri_tmp) > 1:
                prevday_fri_increase = date_fri_tmp[-1] - date_fri_tmp[-2]
            if len(week_fri_tmp) > 1:
                prevweek_fri_increase = week_fri_tmp[-1] - week_fri_tmp[-2]

            rows.append([
                inday_fol_increase, prevday_fol_increase,
                inweek_fol_increase, prevweek_fol_increase,
                inday_fri_increase, prevday_fri_increase,
                inweek_fri_increase, prevweek_fri_increase
            ])

        colnames = [
            'inday_fol_increase', 'prevday_fol_increase',
            'inweek_fol_increase', 'prevweek_fol_increase',
            'inday_fri_increase', 'prevday_fri_increase',
            'inweek_fri_increase', 'prevweek_fri_increase'
        ]
        features = pd.DataFrame(rows, columns=colnames)

        follow_tmp = np.array(follow_tmp)
        features['user_follow_mean'] = follow_tmp.mean()
        features['user_follow_std'] = follow_tmp.std()

        friend_tmp = np.array(friend_tmp)
        features['user_friend_mean'] = friend_tmp.mean()
        features['user_friend_std'] = friend_tmp.std()

        fav_tmp = np.array(fav_tmp)
        features['user_favorite_mean'] = fav_tmp.mean()
        features['user_favorite_std'] = fav_tmp.std()

        features['user_entities_unique'] = len(set(entities_tmp))
        features['user_mentions_unique'] = len(set(mentions_tmp))
        features['user_hashtags_unique'] = len(set(hashtags_tmp))
        features['user_urls_unique'] = len(set(urls_tmp))

        inday_fol_increase_features[user_index] = features['inday_fol_increase']
        prevday_fol_increase_features[user_index] = features['prevday_fol_increase']
        inweek_fol_increase_features[user_index] = features['inweek_fol_increase']
        prevweek_fol_increase_features[user_index] = features['prevweek_fol_increase']

        inday_fri_increase_features[user_index] = features['inday_fri_increase']
        prevday_fri_increase_features[user_index] = features['prevday_fri_increase']
        inweek_fri_increase_features[user_index] = features['inweek_fri_increase']
        prevweek_fri_increase_features[user_index] = features['prevweek_fri_increase']

        user_follow_mean_features[user_index] = features['user_follow_mean']
        user_follow_std_features[user_index] = features['user_follow_std']
        user_friend_mean_features[user_index] = features['user_friend_mean']
        user_friend_std_features[user_index] = features['user_friend_std']
        user_favorite_mean_features[user_index] = features['user_favorite_mean']
        user_favorite_std_features[user_index] = features['user_favorite_std']

        user_entities_unique_features[user_index] = features['user_entities_unique']
        user_mentions_unique_features[user_index] = features['user_mentions_unique']
        user_hashtags_unique_features[user_index] = features['user_hashtags_unique']
        user_urls_unique_features[user_index] = features['user_urls_unique']
    
    X_all['inday_fol_increase'] = inday_fol_increase_features
    X_all['prevday_fol_increase'] = prevday_fol_increase_features
    X_all['inweek_fol_increase'] = inweek_fol_increase_features
    X_all['prevweek_fol_increase'] = prevweek_fol_increase_features

    X_all['inday_fri_increase'] = inday_fri_increase_features
    X_all['prevday_fri_increase'] = prevday_fri_increase_features
    X_all['inweek_fri_increase'] = inweek_fri_increase_features
    X_all['prevweek_fri_increase'] = prevweek_fri_increase_features

    X_all['user_friend_mean'] = user_friend_mean_features
    X_all['user_friend_std'] = user_friend_std_features
    X_all['user_follow_mean'] = user_follow_mean_features
    X_all['user_follow_std'] = user_follow_std_features
    X_all['user_favorite_mean'] = user_favorite_mean_features
    X_all['user_favorite_std'] = user_favorite_std_features

    X_all['user_entities_unique'] = user_entities_unique_features
    X_all['user_mentions_unique'] = user_mentions_unique_features
    X_all['user_hashtags_unique'] = user_hashtags_unique_features
    X_all['user_urls_unique'] = user_urls_unique_features

    for feat in user_stats_features:
        X_train[feat] = X_all.iloc[:len(X_train), :][feat]
        X_valid[feat] = X_all.iloc[len(X_train): len(X_train) + len(X_valid), :][feat]
        X_test[feat] = X_all.iloc[len(X_train) + len(X_valid):, :][feat]

    X_all = pd.concat([X_train, X_valid, X_test], axis=0)

    users = []
    user_entities = []
    user_mentions = []
    user_hashtags = []
    user_urls = []

    user_num = X_all['username'].nunique()

    for idx, (username, X_user) in tqdm(enumerate(X_all.groupby('username')), total=user_num):
        X_user = X_user.sort_values('timestamp', ascending=True)
        user_index = X_user.index.tolist()

        date_fol_tmp = []
        week_fol_tmp = []
        date_fri_tmp = []
        week_fri_tmp = []
        follow_tmp = []
        friend_tmp = []
        fav_tmp = []
        entities_tmp, mentions_tmp, hashtags_tmp, urls_tmp = [], [], [], []
        date_tmp = ''
        week_tmp = ''

        rows = []
        for dt, fol, fri, fav, entities, mentions, hashtags, urls in X_user[['timestamp', '#followers', '#friends', '#favorites', 'entities', 'mentions', 'hashtags', 'urls']].values:
            dt = pd.to_datetime(dt)
            if date_tmp != dt.date():
                date_tmp = dt.date()
                date_fol_tmp.append(fol)
                date_fri_tmp.append(fri)
            if week_tmp != dt.week:
                week_tmp = dt.week
                week_fol_tmp.append(fol)
                week_fri_tmp.append(fri)
            follow_tmp.append(fol)
            friend_tmp.append(fri)
            fav_tmp.append(fav)
            entities_tmp.extend(entities)
            mentions_tmp.extend(mentions)
            hashtags_tmp.extend(hashtags)
            urls_tmp.extend(urls)

            inday_fol_increase = fol - date_fol_tmp[-1]
            inweek_fol_increase = fol - week_fol_tmp[-1]
            inday_fri_increase = fri - date_fri_tmp[-1]
            inweek_fri_increase = fri - week_fri_tmp[-1]

            prevday_fol_increase = 0
            prevweek_fol_increase = 0
            if len(date_fol_tmp) > 1:
                prevday_fol_increase = date_fol_tmp[-1] - date_fol_tmp[-2]
            if len(week_fol_tmp) > 1:
                prevweek_fol_increase = week_fol_tmp[-1] - week_fol_tmp[-2]

            prevday_fri_increase = 0
            prevweek_fri_increase = 0
            if len(date_fri_tmp) > 1:
                prevday_fri_increase = date_fri_tmp[-1] - date_fri_tmp[-2]
            if len(week_fri_tmp) > 1:
                prevweek_fri_increase = week_fri_tmp[-1] - week_fri_tmp[-2]

            rows.append([
                inday_fol_increase, prevday_fol_increase,
                inweek_fol_increase, prevweek_fol_increase,
                inday_fri_increase, prevday_fri_increase,
                inweek_fri_increase, prevweek_fri_increase
            ])

    colnames = [
        'inday_fol_increase', 'prevday_fol_increase',
        'inweek_fol_increase', 'prevweek_fol_increase',
        'inday_fri_increase', 'prevday_fri_increase',
        'inweek_fri_increase', 'prevweek_fri_increase'
    ]
    features = pd.DataFrame(rows, columns=colnames)

    follow_tmp = np.array(follow_tmp)
    features['user_follow_mean'] = follow_tmp.mean()
    features['user_follow_std'] = follow_tmp.std()

    friend_tmp = np.array(friend_tmp)
    features['user_friend_mean'] = friend_tmp.mean()
    features['user_friend_std'] = friend_tmp.std()

    fav_tmp = np.array(fav_tmp)
    features['user_favorite_mean'] = fav_tmp.mean()
    features['user_favorite_std'] = fav_tmp.std()

    features['user_entities_unique'] = len(set(entities_tmp))
    features['user_mentions_unique'] = len(set(mentions_tmp))
    features['user_hashtags_unique'] = len(set(hashtags_tmp))
    features['user_urls_unique'] = len(set(urls_tmp))

    inday_fol_increase_features[user_index] = features['inday_fol_increase']
    prevday_fol_increase_features[user_index] = features['prevday_fol_increase']
    inweek_fol_increase_features[user_index] = features['inweek_fol_increase']
    prevweek_fol_increase_features[user_index] = features['prevweek_fol_increase']

    inday_fri_increase_features[user_index] = features['inday_fri_increase']
    prevday_fri_increase_features[user_index] = features['prevday_fri_increase']
    inweek_fri_increase_features[user_index] = features['inweek_fri_increase']
    prevweek_fri_increase_features[user_index] = features['prevweek_fri_increase']

    user_follow_mean_features[user_index] = features['user_follow_mean']
    user_follow_std_features[user_index] = features['user_follow_std']
    user_friend_mean_features[user_index] = features['user_friend_mean']
    user_friend_std_features[user_index] = features['user_friend_std']
    user_favorite_mean_features[user_index] = features['user_favorite_mean']
    user_favorite_std_features[user_index] = features['user_favorite_std']

    user_entities_unique_features[user_index] = features['user_entities_unique']
    user_mentions_unique_features[user_index] = features['user_mentions_unique']
    user_hashtags_unique_features[user_index] = features['user_hashtags_unique']
    user_urls_unique_features[user_index] = features['user_urls_unique']

    X_all['inday_fol_increase'] = inday_fol_increase_features
    X_all['prevday_fol_increase'] = prevday_fol_increase_features
    X_all['inweek_fol_increase'] = inweek_fol_increase_features
    X_all['prevweek_fol_increase'] = prevweek_fol_increase_features

    X_all['inday_fri_increase'] = inday_fri_increase_features
    X_all['prevday_fri_increase'] = prevday_fri_increase_features
    X_all['inweek_fri_increase'] = inweek_fri_increase_features
    X_all['prevweek_fri_increase'] = prevweek_fri_increase_features

    X_all['user_friend_mean'] = user_friend_mean_features
    X_all['user_friend_std'] = user_friend_std_features
    X_all['user_follow_mean'] = user_follow_mean_features
    X_all['user_follow_std'] = user_follow_std_features
    X_all['user_favorite_mean'] = user_favorite_mean_features
    X_all['user_favorite_std'] = user_favorite_std_features

    X_all['user_entities_unique'] = user_entities_unique_features
    X_all['user_mentions_unique'] = user_mentions_unique_features
    X_all['user_hashtags_unique'] = user_hashtags_unique_features
    X_all['user_urls_unique'] = user_urls_unique_features

    for feat in user_stats_features:
        X_train[feat] = X_all.iloc[:len(X_train), :][feat]
        X_valid[feat] = X_all.iloc[len(X_train): len(X_train) + len(X_valid), :][feat]
        X_test[feat] = X_all.iloc[len(X_train) + len(X_valid):, :][feat]
    
        X_all = pd.concat([
            X_train, X_valid, X_test
        ], axis=0)

    users = []
    user_entities = []
    user_mentions = []
    user_hashtags = []
    user_urls = []

    user_num = X_all['username'].nunique()
    for idx, (username, X_user) in tqdm(enumerate(X_all.groupby('username')), total=user_num):

        users.append(username)

        X_user = X_user.sort_values('timestamp', ascending=True)
        user_index = X_user.index.tolist()

        entities_tmp, mentions_tmp, hashtags_tmp, urls_tmp = [], [], [], []

        rows = []
        for entities, mentions, hashtags, urls in X_user[['entities', 'mentions', 'hashtags', 'urls']].values:

            entities_tmp.extend(entities)
            mentions_tmp.extend(mentions)
            hashtags_tmp.extend(hashtags)
            urls_tmp.extend(urls)

        entities_tmp = [e for e in entities_tmp if e != 'null']
        mentions_tmp = [e for e in mentions_tmp if e != 'null']
        hashtags_tmp = [e for e in hashtags_tmp if e != 'null']
        urls_tmp = [e for e in urls_tmp if e != 'null']

        user_entities.append(' '.join(entities_tmp))
        user_mentions.append(' '.join(mentions_tmp))
        user_hashtags.append(' '.join(hashtags_tmp))
        user_urls.append(' '.join(urls_tmp))
    
    user_features = pd.DataFrame()
    user_features['username'] = users

    features = tfidf_reduce(user_entities, n_gram=n_gram, n_components=dim_red_dim, dr_name=dim_red_name)
    features = features.add_prefix('entities_TFIDF_{}_'.format(dim_red_name))
    user_features = pd.concat([
        user_features, features
    ], axis=1)


    features = tfidf_reduce(user_mentions, n_gram=n_gram, n_components=dim_red_dim, dr_name=dim_red_name)
    features = features.add_prefix('mentions_TFIDF_{}_'.format(dim_red_name))
    user_features = pd.concat([
        user_features, features
    ], axis=1)


    features = tfidf_reduce(user_hashtags, n_gram=n_gram, n_components=dim_red_dim, dr_name=dim_red_name)
    features = features.add_prefix('hashtags_TFIDF_{}_'.format(dim_red_name))
    user_features = pd.concat([
        user_features, features
    ], axis=1)


    features = tfidf_reduce(user_urls, n_gram=n_gram, n_components=dim_red_dim, dr_name=dim_red_name)
    features = features.add_prefix('urls_TFIDF_{}_'.format(dim_red_name))
    user_features = pd.concat([
        user_features, features
    ], axis=1)

    user_features.to_csv(f'{feature_dir}/user_features.csv', index=False)

    X_all = pd.concat([
    X_train, X_valid, X_test
    ], axis=0)

    user_stats_features_list = [
        'user_follow_mean', 'user_follow_std',
        'user_friend_mean', 'user_friend_std',
        'user_favorite_mean', 'user_favorite_std',
        'user_entities_unique', 'user_mentions_unique', 'user_hashtags_unique', 'user_urls_unique'
    ]
    X_stats_train = pd.DataFrame()
    X_stats_valid = pd.DataFrame()
    X_stats_test = pd.DataFrame()
    X_stats_train = pd.concat([
        X_stats_train,  X_train[user_stats_features_list]
    ], axis=1)
    X_stats_valid = pd.concat([
        X_stats_valid, X_valid[user_stats_features_list]
    ], axis=1)
    X_stats_test = pd.concat([
        X_stats_test, X_test[user_stats_features_list]
    ], axis=1)
    X_stats_all = pd.concat([
        X_stats_train, X_stats_valid, X_stats_test
    ], axis=0)
    X_stats_all = pd.concat([
        X_all[['username']], X_stats_all
    ], axis=1)
    user_stats_features = X_stats_all.groupby('username').head(1)

    user_topic_features = pd.read_csv(f'{feature_dir}/user_features.csv')
    user_stats_topic_features = pd.merge(user_stats_features, user_topic_features, on='username')

    #APPLYING K MEANS
    #CLUSTERING USER STATS
    USER_CLUSTER_NUM = 1000
    user_stats_features, user_to_stats_cluster = apply_kmeans(user_stats_features, USER_CLUSTER_NUM)
    X_train[f'user_stats_cluster_{USER_CLUSTER_NUM}'] = X_train['username'].map(user_to_stats_cluster)
    X_valid[f'user_stats_cluster_{USER_CLUSTER_NUM}'] = X_valid['username'].map(user_to_stats_cluster)
    X_test[f'user_stats_cluster_{USER_CLUSTER_NUM}'] = X_test['username'].map(user_to_stats_cluster)

    user_topic_features, user_to_topic_cluster = apply_kmeans(user_topic_features, USER_CLUSTER_NUM)
    X_train[f'user_topic_cluster_{USER_CLUSTER_NUM}'] = X_train['username'].map(user_to_topic_cluster)
    X_valid[f'user_topic_cluster_{USER_CLUSTER_NUM}'] = X_valid['username'].map(user_to_topic_cluster)
    X_test[f'user_topic_cluster_{USER_CLUSTER_NUM}'] = X_test['username'].map(user_to_topic_cluster)

    user_stats_topic_features, user_to_stats_topic_cluster = apply_kmeans(user_stats_topic_features, USER_CLUSTER_NUM)
    X_train[f'user_stats_topic_cluster_{USER_CLUSTER_NUM}'] = X_train['username'].map(user_to_stats_topic_cluster)
    X_valid[f'user_stats_topic_cluster_{USER_CLUSTER_NUM}'] = X_valid['username'].map(user_to_stats_topic_cluster)
    X_test[f'user_stats_topic_cluster_{USER_CLUSTER_NUM}'] = X_test['username'].map(user_to_stats_topic_cluster)

    for feat in dense_features:
        X_train[feat] = X_train[feat].fillna(0.0)
        X_valid[feat] = X_valid[feat].fillna(0.0)
        X_test[feat] = X_test[feat].fillna(0.0)
        tmp = pd.concat([
            X_train[feat], X_valid[feat], X_test[feat]
        ])
        max_v, min_v = tmp.max(), tmp.min()
        print(f'{feat} - MMS Scaling > max: {max_v}, min: {min_v}')
        X_train[feat] = (X_train[feat] - min_v) / (max_v - min_v)
        X_valid[feat] = (X_valid[feat] - min_v) / (max_v - min_v)
        X_test[feat] = (X_test[feat] - min_v) / (max_v - min_v)
        save_as_feather(feat, feature_dir, X_train, X_valid, X_test)

    for feat in sparse_features:
        if feat in user_features:
            X_train, X_valid, X_test, key2index, unique_num = label_encording_threshold(
                feat, X_train, X_valid, X_test, low_freq_th=1
            )
            save_as_feather(feat, feature_dir, X_train, X_valid, X_test)
        else:
            X_train, X_valid, X_test, unique_num = label_encording(
                feat, X_train, X_valid, X_test
            )
            save_as_feather(feat, feature_dir, X_train, X_valid, X_test)

    for feat in varlen_ori_features:
        X_train, X_valid, X_test, key2index, unique_num = varlen_label_encording_threshold(
            feat, X_train, X_valid, X_test, low_freq_th=1
        )
        save_as_feather(feat, feature_dir, X_train, X_valid, X_test)

    

def apply_kmeans(df, cluster_num):

  kmeans = KMeans(n_clusters=cluster_num, random_state=45)
  kmeans.fit(df.iloc[:, 1:])
  df['kmeans'] = kmeans.labels_
  user_to_cluster = {i: j for i, j in df[['username', 'kmeans']].values}
  return df, user_to_cluster

def main(input_dir, feature_dir, feature_list):

    X_train = pd.read_csv(f'{input_dir}/train.csv', nrows = 100000)
    X_train.columns = feature_list
    
    X_valid = pd.read_csv(f'{input_dir}/val.csv', nrows = 100000)
    X_valid.columns = feature_list
    
    X_test = pd.read_csv(f'{input_dir}/test.csv', nrows = 100000)
    X_test.columns = feature_list
    
    y_train = pd.read_csv(f'{input_dir}/train_labels.csv', header=None, nrows=100000, skiprows=1).T.values[0].reshape(-1, 1)
    
    dense_features, sparse_features, varlen_sparse_features, to_encode_count, to_encode_target,\
         user_stats_features, time_cat_features, time_num_features, tweet_metrics_features, tweet_metrics_log_features,\
              tweet_metrics_bin_features = init_features()
    
    preprocess(feature_dir, X_train, X_valid, X_test, time_cat_features, time_num_features, tweet_metrics_features,\
        tweet_metrics_log_features, tweet_metrics_bin_features, to_encode_count, y_train, to_encode_target, varlen_sparse_features, \
            user_stats_features, dense_features, sparse_features)
    

if __name__==__main__:
    input_dir = '/content/drive/MyDrive/dataset/data/'
    feature_dir = '/content/drive/MyDrive/dataset/feat_new/'
    feature_list  = ["tweetId", "username", "timestamp", "#followers", "#friends", "#retweets", "#favorites", "entities", "sentiment", "mentions", "hashtags", "urls"]

    main(input_dir, feature_dir, feature_list)



