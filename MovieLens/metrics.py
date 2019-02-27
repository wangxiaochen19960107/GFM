import keras
import math 
import numpy as np

class metric_mf(keras.callbacks.Callback):
    def __init__(self,test_users_id, test_items_id, hit_score, ndcg_score):   
        self.test_users_id = test_users_id
        self.test_items_id = test_items_id
        self.K = 10
        self.hit_score = hit_score
        self.ndcg_score = ndcg_score
 
    def on_epoch_end(self, epoch, logs={}):    
        hit_rate = 0
        ndcg_value = 0 
        #用户数量
        users_num = self.test_users_id[-1]
        #使用模型进行预测
        predictions = self.model.predict([self.test_users_id, self.test_items_id])
        
        for i in range(1, users_num+1):
            pred_temp = predictions[(i-1)*100: i*100]
            #预测结果，真正例预测结果在最后一个
            neg_predict, pos_predict = pred_temp[:-1], pred_temp[-1]
            #判断预测值比真正例大的有没有超过K个
            position = (neg_predict >= pos_predict).sum()
            hit = position < self.K
            if hit:
                ndcg = math.log(2) / math.log(position+2)     
                hit_rate += 1
                ndcg_value += ndcg 
        
        hit_rate /= users_num
        ndcg_value /= users_num
        self.hit_score.append(round(hit_rate, 4))
        self.ndcg_score.append(round(ndcg_value, 4))
        print('hit@10: ' + str(hit_rate) + ' , ndcg@10: ' + str(ndcg_value))
        return



class metric_deepfm(keras.callbacks.Callback):
    def __init__(self,test_users_id, test_items_id, test_sexes, test_ages, test_jobs, test_genres, hit_score, ndcg_score):   
        self.test_users_id = test_users_id
        self.test_items_id = test_items_id
        self.test_sexes = test_sexes
        self.test_ages = test_ages
        self.test_jobs = test_jobs
        self.test_genres = test_genres
        self.K = 10
        self.hit_score = hit_score
        self.ndcg_score = ndcg_score
 
    def on_epoch_end(self, epoch, logs={}):    
        hit_rate = 0
        ndcg_value = 0 
        #用户数量
        users_num = self.test_users_id[-1]
        #使用模型进行预测
        predictions = self.model.predict([self.test_users_id, self.test_items_id, self.test_sexes,\
            self.test_ages, self.test_jobs, self.test_genres])
        
        for i in range(1, users_num+1):
            pred_temp = predictions[(i-1)*100: i*100]
            #预测结果，真正例预测结果在最后一个
            neg_predict, pos_predict = pred_temp[:-1], pred_temp[-1]
            #判断预测值比真正例大的有没有超过K个
            position = (neg_predict >= pos_predict).sum()
            hit = position < self.K
            if hit:
                ndcg = math.log(2) / math.log(position+2)     
                hit_rate += 1
                ndcg_value += ndcg 
        
        hit_rate /= users_num
        ndcg_value /= users_num
        self.hit_score.append(round(hit_rate, 4))
        self.ndcg_score.append(round(ndcg_value, 4))
        print('hit@10: ' + str(hit_rate) + ' , ndcg@10: ' + str(ndcg_value))
        return
 

class metric_deepfm_gbdt(keras.callbacks.Callback):
    def __init__(self, data, hit_score, ndcg_score):     
        self.data = data
        self.K = 10
        self.hit_score = hit_score
        self.ndcg_score = ndcg_score
 
    def on_epoch_end(self, epoch, logs={}):    
        hit_rate = 0
        ndcg_value = 0 
        #用户数量
        users_num = int(len(self.data[0])/100)
        
        #使用模型进行预测
        predictions = self.model.predict(self.data)
                        
        for i in range(1, users_num+1):
            pred_temp = predictions[(i-1)*100: i*100]

            #预测结果，真正例预测结果在最后一个
            neg_predict, pos_predict = pred_temp[:-1], pred_temp[-1]
            #判断预测值比真正例大的有没有超过K个
            position = (neg_predict >= pos_predict).sum()
            hit = position < self.K
            if hit:
                ndcg = math.log(2) / math.log(position+2)     
                hit_rate += 1
                ndcg_value += ndcg 
        
        hit_rate /= users_num
        ndcg_value /= users_num
        self.hit_score.append(round(hit_rate, 4))
        self.ndcg_score.append(round(ndcg_value, 4))
        print('hit@10: ' + str(hit_rate) + ' , ndcg@10: ' + str(ndcg_value))
        return


def metric_gbdt(y_pred, users_num):
    hit_rate = 0
    ndcg_value = 0 
    
    for i in range(1, users_num+1):
        pred = y_pred[(i-1)*100: i*100]
        neg_predict, pos_predict = pred[:-1], pred[-1]
        position = (neg_predict >= pos_predict).sum()
        hit = position < 10
        if hit:
            ndcg = math.log(2) / math.log(position+2)     
            hit_rate += 1
            ndcg_value += ndcg 

    hit_rate /= users_num
    ndcg_value /= users_num
    print('hit@10: ' + str(round(hit_rate, 4)) + ' , ndcg@10: ' + str(round(ndcg_value, 4)))

    return
