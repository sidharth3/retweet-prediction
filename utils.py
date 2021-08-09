from datetime import datetime, timezone
import math
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
import pandas as pd
from sklearn.preprocessing import LabelEncoder
#from tqdim import tqdm
import numpy as np
from sklearn.preprocessing import MinMaxScaler

import torch
import torch.nn as nn
import torch.nn.functional as F

def week_of_month(dt):
    """ Returns the week of the month for the specified date.
    """
    first_day = dt.replace(day=1)

    dom = dt.day
    adjusted_dom = dom + first_day.weekday()

    return int(math.ceil(adjusted_dom / 7.0))


def extract_time_features(df):
    timestamp = []
    for x in df['timestamp'].values:
        timestamp.append(datetime.strptime(' '.join(x.split(' ')[1:]), '%b %d %H:%M:%S %z %Y'))
    weekday = [ts.weekday() for ts in timestamp]
    hour = [ts.hour for ts in timestamp]
    day = [ts.day for ts in timestamp]
    wom = [week_of_month(ts) for ts in timestamp]

    latest = datetime(2020, 6, 1, 0, 0, 0, 0, timezone.utc)
    diff_from_latest = [(latest - ts).seconds for ts in timestamp]

    df['weekday'] = weekday
    df['hour'] = hour
    df['day'] = day
    df['week_of_month'] = wom
    df['diff_from_latest'] = diff_from_latest

    return df

def save_feature(feat, save_dir, X_train, X_valid, X_test):
    
    X_train[[feat]].reset_index(drop=True).to_feather(f'{save_dir}/{feat}_train.feather')
    X_valid[[feat]].reset_index(drop=True).to_feather(f'{save_dir}/{feat}_valid.feather')
    X_test[[feat]].reset_index(drop=True).to_feather(f'{save_dir}/{feat}_test.feather')
    
    print(f'Saving [{feat}] to -> save_dir ... [{save_dir}]')

def qbin_transform(colname, X_train, X_valid, X_test, nbin):

    df = pd.concat([
        X_train, X_valid, X_test
    ], axis=0)

    s_cut, _ = pd.qcut(df[colname], nbin, retbins=True, duplicates='drop')
    # s_cut, _ = pd.qcut(df[colname], nbin, retbins=True)
    label_enc = LabelEncoder()
    label_enc.fit(s_cut)
    s_cut = label_enc.transform(s_cut)

    X_train[f'{colname}_qbin_{nbin}'] = s_cut[:len(X_train)]
    X_valid[f'{colname}_qbin_{nbin}'] = s_cut[len(X_train):len(X_train) + len(X_valid)]
    X_test[f'{colname}_qbin_{nbin}'] = s_cut[len(X_train) + len(X_valid):]

    return X_train, X_valid, X_test

def split_sentiment(df):
    senti = df['sentiment'].values
    senti = [i.split() for i in senti]
    senti_pos = [int(i[0]) for i in senti]
    senti_neg = [int(i[1]) for i in senti]
    df['sentiment_pos'] = senti_pos
    df['sentiment_neg'] = senti_neg
    return df

def preprocess_entities(df):
    values = [e.split(';')[:-1] for e in df['entities'].values]
    rows = []
    for x in values:
        if x[0] == 'null':
            row = ['null']
        else:
            row = []
            for e in x:
                e = e.split(':')
                row.append(e[1])
        rows.append(row)
    df['entities'] = rows
    return df

def preprocess_varlen(df, colname):
    values = df[colname].values
    rows = []
    for x in values:
        if x == "null;":
            row = ['null']
        else:
            row = []
            for m in x.split():
                row.append(m)
        rows.append(row)
    df[colname] = rows
    return df


def feature_count(x):
    if len(x) == 1 and x[0] == 'null':
        return 0
    return len(x)

def tfidf_reduce(sentences, n_gram, n_components, dr_name):
    tfv = TfidfVectorizer(min_df=1, max_features=None,
                          strip_accents='unicode', analyzer='word', token_pattern=r'(?u)\b\w+\b',
                          ngram_range=(1, n_gram), use_idf=1, smooth_idf=1, sublinear_tf=1)
    if dr_name == "svd":
        dr_model = TruncatedSVD(n_components=n_components, random_state=1337)

    tfidf_col = tfv.fit_transform(sentences)
    dr_col = dr_model.fit_transform(tfidf_col)
    dr_col = pd.DataFrame(dr_col)
    return dr_col

def extract_text_tfidf(X_train, X_valid, X_test, n_gram=1, dr_name="svd", dr_dim=5):
    X_all = pd.concat([
        X_train, X_valid, X_test
    ], axis=0)

    tweet_url_publisher = []
    for urls in X_all['urls'].values:
        publisher = ''
        if not urls[0] == 'null':
            for url in urls:
                if 'twitter.com' in url:
                    url = url.split('/')
                    if len(url) > 3:
                        publisher = url[3]
        tweet_url_publisher.append(publisher)

    X_all['tweet_url_publisher'] = tweet_url_publisher

    texts = []
    for m, e, h, tup in X_all[['mentions', 'entities', 'hashtags', 'tweet_url_publisher']].values:
        if len(m) == 0 or m[0] == 'null':
            m = []
        if len(e) == 0 or e[0] == 'null':
            e = []
        if len(h) == 0 or h[0] == 'null':
            h = []

        # factorize entities
        e_div = []
        for ent in e:
            e_div.append(ent)
            ent_div = ent.split('_')
            if len(ent_div) > 1:
                e_div.extend(ent_div)
        text = m + e_div + h + [tup]
        texts.append(' '.join(text))

    features = tfidf_reduce(texts, n_gram=n_gram, n_components=dr_dim, dr_name=dr_name)
    features = features.add_prefix('TFIDF_{}_'.format(dr_name))

    X_train = pd.concat([
        X_train, features.iloc[:len(X_train), :].reset_index(drop=True)
    ], axis=1)
    X_valid = pd.concat([
        X_valid, features.iloc[len(X_train):len(X_train) + len(X_valid), :].reset_index(drop=True)
    ], axis=1)
    X_test = pd.concat([
        X_test, features.iloc[len(X_train) + len(X_valid):, :].reset_index(drop=True)
    ], axis=1)

    return X_train, X_valid, X_test

def label_scaling(val):
    val = np.log(val + 1)
    scaler = MinMaxScaler()
    scaler.fit(val)
    val = scaler.transform(val)
    return scaler, val


def label_inverse_scaling(scaler, val):
    val = scaler.inverse_transform(val)
    val = np.exp(val) - 1
    return val

def user_encoding(colname, X_train, X_valid, X_test, low_freq_th):
    tmp = pd.Series()
    tmp = pd.concat([tmp, X_train[colname]])
    tmp = pd.concat([tmp, X_valid[colname]])
    tmp = pd.concat([tmp, X_test[colname]])

    count = tmp.value_counts()
    low_freq_list = count[count <= low_freq_th].index.tolist()
    count = count[count > low_freq_th]

    key2index = {i: 0 for i in low_freq_list}
    idx = 1
    for i in count.index:
        key2index[i] = idx
        idx += 1
    unique_num = idx + 1

    X_train[colname] = X_train[colname].map(key2index)
    X_valid[colname] = X_valid[colname].map(key2index)
    X_test[colname] = X_test[colname].map(key2index)

    print(f'{colname} - Num. of use as low frequency category: {len(low_freq_list)}')
    print(f'{colname} - Num. of unique category: {unique_num}')

    return X_train, X_valid, X_test, key2index, unique_num


def label_encoding(colname, X_train, X_valid, X_test):
    
    label_enc = LabelEncoder()
    tmp = pd.concat([
        X_train[colname], X_valid[colname], X_test[colname]])
    label_enc.fit(tmp)
    X_train[colname] = label_enc.transform(X_train[colname])
    X_valid[colname] = label_enc.transform(X_valid[colname])
    X_test[colname] = label_enc.transform(X_test[colname])
    return X_train, X_valid, X_test, len(label_enc.classes_)

def varlen_encoding(colname, X_train, X_valid, X_test, low_freq_th):

    entities = []
    for x in X_train[colname].values:
        entities.extend(x)
    for x in X_valid[colname].values:
        entities.extend(x)
    for x in X_test[colname].values:
        entities.extend(x)
    entities = pd.Series(entities)

    count = entities.value_counts()
    low_freq_list = count[count <= low_freq_th].index.tolist()
    count = count[count > low_freq_th]

    key2index = {i: 1 for i in low_freq_list}  # null: 0, low_freq: 1, ...
    idx = 2
    for i in count.index:
        key2index[i] = idx
        idx += 1
    unique_num = idx + 1

    X_train[colname] = X_train[colname].apply(map_varlen, dict=key2index)
    X_valid[colname] = X_valid[colname].apply(map_varlen, dict=key2index)
    X_test[colname] = X_test[colname].apply(map_varlen, dict=key2index)

    print(f'{colname} - Num. of use as low frequency category: {len(low_freq_list)}')
    print(f'{colname} - Num. of unique category: {unique_num}')

    return X_train, X_valid, X_test, key2index, unique_num

def encode_varlen_count(feat, X_train, X_valid, X_test):

    X_all = pd.concat([
        X_train, X_valid, X_test
    ], axis=0)

    counts = {}
    for varlen_list in X_train[feat].values:
        if len(varlen_list) == 0:
            continue
        if varlen_list[0] == 'null':
            continue
        for x in varlen_list:
            if x not in counts:
                counts[x] = 0
            counts[x] += 1

    varlen_ce = []
    for varlen_list in X_all[feat].values:
        if len(varlen_list) == 0:
            varlen_ce.append(0)
            continue
        if varlen_list[0] == 'null':
            varlen_ce.append(0)
            continue
        val = 0
        for x in varlen_list:
            if x in counts:
                val += counts[x]
            else:
                val += 0
        varlen_ce.append(val)

    X_train[f'{feat}_ce'] = varlen_ce[:len(X_train)]
    X_valid[f'{feat}_ce'] = varlen_ce[len(X_train):len(X_train) + len(X_valid)]
    X_test[f'{feat}_ce'] = varlen_ce[len(X_train) + len(X_valid):]

    return X_train, X_valid, X_test


def encode_varlen_target(feat, X_train, X_valid, X_test, y_train, fold):

    cat_target_mean = {}

    varlen_te_train = np.zeros(len(X_train))
    for fold_idx, (trn_idx, val_idx) in enumerate(fold.split(X_train)):
        X_fold_tra = X_train.iloc[trn_idx, :]
        cat_indexes = {}
        for i, varlen_list in enumerate(X_fold_tra[feat].values):
            if len(varlen_list) == 0:
                continue
            if varlen_list[0] == 'null':
                continue
            for x in varlen_list:
                if x not in cat_indexes:
                    cat_indexes[x] = []
                cat_indexes[x].append(i)

        for x, index in cat_indexes.items():
            if x not in cat_target_mean:
                cat_target_mean[x] = {i: 0.0 for i in range(fold.n_splits)}
            cat_target_mean[x][fold_idx] = y_train[index].mean()

        for i in val_idx:
            varlen_list = X_train[feat].values[i]
            val = 0
            for x in varlen_list:
                if x in cat_target_mean:
                    val += cat_target_mean[x][fold_idx]
                else:
                    val += 0.0
            varlen_te_train[i] = val
    X_train[f'{feat}_te'] = varlen_te_train

    varlen_te = []
    for varlen_list in X_valid[feat].values:
        if len(varlen_list) == 0:
            varlen_te.append(0.0)
            continue
        if varlen_list[0] == 'null':
            varlen_te.append(0.0)
            continue
        val = 0
        for x in varlen_list:
            if x in cat_target_mean:
                for fold_idx in range(fold.n_splits):
                    val += cat_target_mean[x][fold_idx]
                val /= fold.n_splits
            else:
                val += 0.0
        varlen_te.append(val)
    X_valid[f'{feat}_te'] = varlen_te

    varlen_te = []
    for varlen_list in X_test[feat].values:
        if len(varlen_list) == 0:
            varlen_te.append(0.0)
            continue
        if varlen_list[0] == 'null':
            varlen_te.append(0.0)
            continue
        val = 0
        for x in varlen_list:
            if x in cat_target_mean:
                for fold_idx in range(fold.n_splits):
                    val += cat_target_mean[x][fold_idx]
                val /= fold.n_splits
            else:
                val += 0.0
        varlen_te.append(val)
    X_test[f'{feat}_te'] = varlen_te

    return X_train, X_valid, X_test
    
def map_varlen(x, dict):
    return [dict[i] for i in x]



class SequencePoolingLayer(nn.Module):
    """The SequencePoolingLayer is used to apply pooling operation(sum,mean,max) on variable-length sequence feature/multi-value feature.
      Input shape
        - A list of two  tensor [seq_value,seq_len]
        - seq_value is a 3D tensor with shape: ``(batch_size, T, embedding_size)``
        - seq_len is a 2D tensor with shape : ``(batch_size, 1)``,indicate valid length of each sequence.
      Output shape
        - 3D tensor with shape: ``(batch_size, 1, embedding_size)``.
      Arguments
        - **mode**:str.Pooling operation to be used,can be sum,mean or max.
    """

    def __init__(self, mode='mean', supports_masking=False, device='cpu'):

        super(SequencePoolingLayer, self).__init__()
        if mode not in ['sum', 'mean', 'max']:
            raise ValueError('parameter mode should in [sum, mean, max]')
        self.supports_masking = supports_masking
        self.mode = mode
        self.device = device
        self.eps = torch.FloatTensor([1e-8]).to(device)
        self.to(device)

    def _sequence_mask(self, lengths, maxlen=None, dtype=torch.bool):
        # Returns a mask tensor representing the first N positions of each cell.
        if maxlen is None:
            maxlen = lengths.max()
        row_vector = torch.arange(0, maxlen, 1).to(self.device)
        matrix = torch.unsqueeze(lengths, dim=-1)
        mask = row_vector < matrix

        mask.type(dtype)
        return mask

    def forward(self, seq_value_len_list):
        if self.supports_masking:
            uiseq_embed_list, mask = seq_value_len_list  # [B, T, E], [B, 1]
            mask = mask.float()
            user_behavior_length = torch.sum(mask, dim=-1, keepdim=True)
            mask = mask.unsqueeze(2)
        else:
            uiseq_embed_list, user_behavior_length = seq_value_len_list  # [B, T, E], [B, 1]
            mask = self._sequence_mask(user_behavior_length, maxlen=uiseq_embed_list.shape[1],
                                       dtype=torch.float32)  # [B, 1, maxlen]
            mask = torch.transpose(mask, 1, 2)  # [B, maxlen, 1]

        embedding_size = uiseq_embed_list.shape[-1]

        mask = torch.repeat_interleave(mask, embedding_size, dim=2)  # [B, maxlen, E]

        if self.mode == 'max':
            hist = uiseq_embed_list - (1 - mask) * 1e9
            hist = torch.max(hist, dim=1, keepdim=True)[0]
            return hist
        hist = uiseq_embed_list * mask.float()
        hist = torch.sum(hist, dim=1, keepdim=False)

        if self.mode == 'mean':
            hist = torch.div(hist, user_behavior_length.type(torch.float32) + self.eps)

        hist = torch.unsqueeze(hist, dim=1)
        return hist

def get_varlen_pooling_list(embedding_dict, features, feature_index, varlen_sparse_feature_columns, mode_list, device):
    varlen_sparse_embedding_list = []
    for feat in varlen_sparse_feature_columns:
        print(feat)
        for mode in mode_list:
            seq_emb = embedding_dict[f'{feat}__{mode}'](
                features[:, feature_index[feat][0]:feature_index[feat][1]].long()
            )
            #print(seq_emb)
            seq_mask = features[:, feature_index[feat][0]: feature_index[feat][1]].long() != 0

            emb = SequencePoolingLayer(
                mode=mode,
                supports_masking=True,
                device=device
            )([seq_emb, seq_mask])
            varlen_sparse_embedding_list.append(emb)

    return varlen_sparse_embedding_list