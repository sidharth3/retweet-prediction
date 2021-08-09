# Import Libaries
# import the necessary files and libraries
# import the necessary libraries
from sklearn.linear_model import LogisticRegression
from tqdm import tqdm
import preprocessor as p
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import KMeans
import csv 
import pandas as pd
from io import BytesIO
from tensorflow import keras
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
import tensorflow as tf
from sklearn.model_selection import KFold
import logging
# from xfeat import TargetEncoder, CountEncoder
from sklearn.cluster import KMeans
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from scipy.stats import norm
from scipy.stats import zscore
from tqdm import tqdm
import io
import re
import string
import pickle
import pkg_resources
import ast
import ssl
from tensorflow.keras.models import model_from_json

class LinearRegressionModel(torch.nn.Module):

    def __init__(self):
        super(LinearRegressionModel, self).__init__()
        self.linear = nn.Linear(7, 32)  # One in and one out
        self.linear2 = nn.Linear(32, 64)
        self.linear3 = nn.Linear(64,32)
        self.linear4 = nn.Linear(32,1)

    def forward(self, x):
        x= self.linear(x)
        x = nn.functional.relu(x)
        x = self.linear2(x)
        x = nn.functional.relu(x)
        x = self.linear3(x)
        x = nn.functional.relu(x)
        x = self.linear4(x)
        return x

def call_model(response_tweetID, response_username, response_followers, response_friends, response_favorites, response_entities, response_POSsentiment, response_NEGsentiment, response_mentions, response_hashtags):
    lst = [int(response_followers), int(response_friends), int(response_favorites), int(response_POSsentiment), int(response_NEGsentiment), int(response_mentions), int(response_hashtags)]
    print(lst)
    dict = {'#Followers': [response_followers], '#Friends': [response_friends], '#Favorites': [response_favorites], '#Positive_Sentiment': [response_POSsentiment], 'Negative_Sentiment': [response_NEGsentiment], 'No_of_mentions': [response_mentions], 'No_of_hashtags': [response_hashtags]} 
    df = pd.DataFrame(dict)
    model = torch.load("single_pred_num.pth", map_location='cpu')
    label = model(torch.Tensor(lst))
    label = label.item()
    return round(label)