import numpy as np
import pandas as pd
import gc

from keras import layers
from keras.layers import Reshape
from keras import Input
from keras.models import Model
from keras.regularizers import l2 as l2_reg


from preparing import csv_processing
from mylayers import MyMeanPool, MySumLayer, MyFlatten
from metrics import metric_deepfm_gbdt

import keras.backend.tensorflow_backend as KTF
import tensorflow as tf
import os


#GPU动态内存
os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"

config = tf.ConfigProto()
config.gpu_options.allow_growth=True   #不全部占满显存, 按需分配
sess = tf.Session(config=config)

KTF.set_session(sess)


print("开始读取数据")

train_data = []
test_data = []
#读取和处理GBDT编码的训练集
train_trees_df = pd.read_csv('./data/train_gbdt_20.csv', engine='python')
#所有列名
tree_columns = list(train_trees_df.columns)
for col in tree_columns:
    train_trees_col = (np.array(train_trees_df[col].values)).reshape(len(train_trees_df),1)
    train_data.append(train_trees_col)
del train_trees_df
gc.collect()


#读取和处理GBDT编码的测试集
test_trees_df = pd.read_csv('./data/test_gbdt_20.csv', engine='python')
for col in tree_columns:
    test_trees_col = (np.array(test_trees_df[col].values)).reshape(len(test_trees_df),1)
    test_data.append(test_trees_col)
del test_trees_df
gc.collect()


#读取和处理训练集，测试集
train_df = pd.read_csv('./data/train.csv', engine='python')
train_users_id, train_items_id, train_sexes, train_ages, train_jobs, train_genres, train_probs = csv_processing(train_df)

train_data.append(train_users_id)
train_data.append(train_items_id)
train_data.append(train_sexes)
train_data.append(train_ages)
train_data.append(train_jobs)
train_data.append(train_genres)

test_df = pd.read_csv('./data/test.csv', engine='python')
test_users_id, test_items_id, test_sexes, test_ages, test_jobs, test_genres = csv_processing(test_df)
test_data.append(test_users_id)
test_data.append(test_items_id)
test_data.append(test_sexes)
test_data.append(test_ages)
test_data.append(test_jobs)
test_data.append(test_genres)

#用户最大编号，商品最大编号，职业最大编号，电影类型最大编号
user_id_num = max(train_df['user_id'])
item_id_num = max(train_df['item_id'])
age_num = max(train_df['age'])
job_num = max(train_df['job'])
item_genres_num = int(max(max(train_df['genres0']),max(train_df['genres1']),max(train_df['genres2']),max(train_df['genres3']),max(train_df['genres4']),max(train_df['genres5'])))

del train_df
del test_df
gc.collect()


print("数据读取完成")



hit_score = []
ndcg_score = []

l2_reg_1d = l2_reg(0.0086)
l2_reg_kd = l2_reg(0.005)
########################################模型定义########################################
input_cols = []
tree_embed_cols = []
#输入部分
for col in tree_columns:
    input_fea = Input(shape=(1,), name=col)
    input_cols.append(input_fea)

    #深度为5的树节点数最多为2^6 - 1 = 63
    input_fea_embedding_kd = layers.Embedding(63, 32, embeddings_regularizer=l2_reg(0.003))(input_fea)
    input_fea_embedding = layers.Dense(1, activation='sigmoid')(input_fea_embedding_kd)
    tree_embed_cols.append(input_fea_embedding)

user_id_input = Input(shape=[1],name='user_id')
item_id_input = Input(shape=[1],name='item_id')
sex_input = Input(shape=[1],name='sex')
age_input = Input(shape=[1],name='age')
job_input = Input(shape=[1],name='job')
item_genres_input = Input(shape=[6],name='item_genres')
input_cols.append(user_id_input)
input_cols.append(item_id_input)
input_cols.append(sex_input)
input_cols.append(age_input)
input_cols.append(job_input)
input_cols.append(item_genres_input)


#树特征整合
tree_attention = layers.Embedding(user_id_num+1, len(tree_columns), name='attention')(user_id_input)
tree_embed_cols = layers.Concatenate(axis=-1)(tree_embed_cols)
tree_prob = layers.Multiply()([tree_attention,tree_embed_cols])
tree_prob =  Reshape([1])(layers.Dense(1, activation='relu')(tree_prob))

#1d特征相加
embed_user_id = Reshape([1])(layers.Embedding(user_id_num+1, 1)(user_id_input))
embed_item_id = Reshape([1])(layers.Embedding(item_id_num+1, 1)(item_id_input))
embed_sex = Reshape([1])(layers.Embedding(2, 1, embeddings_regularizer=l2_reg_1d)(sex_input))
embed_age = Reshape([1])(layers.Embedding(age_num+1, 1, embeddings_regularizer=l2_reg_1d)(age_input))
embed_job = Reshape([1])(layers.Embedding(job_num+1, 1, embeddings_regularizer=l2_reg_1d)(job_input))
embed_item_genres_ = layers.Embedding(item_genres_num+1, 1, mask_zero = True, embeddings_regularizer=l2_reg_1d)(item_genres_input)
dense_item_genres = MyMeanPool(axis=1)(embed_item_genres_)

y_first_order = layers.Add()([embed_user_id, embed_item_id, embed_sex, embed_age, embed_job, dense_item_genres])


#kd特征交互
latent = 32
embed_user_id_kd = layers.Embedding(user_id_num+1, latent)(user_id_input)
embed_item_id_kd = layers.Embedding(item_id_num+1, latent)(item_id_input)
embed_sex_kd = layers.Embedding(3, latent, embeddings_regularizer=l2_reg_kd)(sex_input)
embed_age_kd = layers.Embedding(age_num+1, latent, embeddings_regularizer=l2_reg_kd)(age_input)
embed_job_kd = layers.Embedding(job_num+1, latent, embeddings_regularizer=l2_reg_kd)(job_input)
embed_item_genres_kd_ = layers.Embedding(item_genres_num+1, latent, mask_zero = True, embeddings_regularizer=l2_reg_kd)(item_genres_input)
embed_item_genres_kd = layers.RepeatVector(1)(MyMeanPool(axis=1)(embed_item_genres_kd_))

embed_kd = layers.Concatenate(axis=1)([embed_user_id_kd, embed_item_id_kd, embed_sex_kd, embed_age_kd, embed_job_kd, embed_item_genres_kd])


#计算
summed_features_emb = MySumLayer(axis=1)(embed_kd) 
summed_features_emb_square = layers.Multiply()([summed_features_emb,summed_features_emb]) 

squared_features_emb = layers.Multiply()([embed_kd, embed_kd])
squared_sum_features_emb = MySumLayer(axis=1)(squared_features_emb) 

sub = layers.Subtract()([summed_features_emb_square, squared_sum_features_emb])

y_second_order = MySumLayer(axis=1)(sub) 


prob = layers.Concatenate(axis=1)([y_first_order, y_second_order, tree_prob])
prob = layers.Dense(1, activation='sigmoid')(prob)


#开始训练
model = Model(input_cols, prob)
model.compile(optimizer='rmsprop',loss='binary_crossentropy')
history = model.fit(train_data, train_probs, epochs=100,batch_size=32768, \
	callbacks = [metric_deepfm_gbdt(test_data, hit_score, ndcg_score)])
    
model.save('./model/model.h5')

print(hit_score)
print(ndcg_score)
