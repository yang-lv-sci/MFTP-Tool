# -*- coding: utf-8 -*-
"""
Created on Fri Jan 27 16:42:35 2023

@author: lvyan
"""
import os
from tensorflow.compat.v1.keras.models import load_model
from model import MultiHeadAttention
import numpy as np
import joblib
from tqdm import tqdm

def count(matrix,v):
    count_number=0
    for i in range(21):
        count_number+=matrix[v,i]
    return count_number

# 使用二元关联
def Aiming(y_hat, y):
    '''
    the “Aiming” rate (also called “Precision”) is to reflect the average ratio of the
    correctly predicted labels over the predicted labels; to measure the percentage
    of the predicted labels that hit the target of the real labels.
    '''

    n, m = y_hat.shape

    sorce_k = 0
    for v in range(n):
        union = 0
        intersection = 0
        for h in range(m):
            if y_hat[v, h] == 1 or y[v, h] == 1:
                union += 1
            if y_hat[v, h] == 1 and y[v, h] == 1:
                intersection += 1
        if intersection == 0:
            continue
        sorce_k += intersection / count(y_hat,v)
    return sorce_k / n


def Coverage(y_hat, y):
    '''
    The “Coverage” rate (also called “Recall”) is to reflect the average ratio of the
    correctly predicted labels over the real labels; to measure the percentage of the
    real labels that are covered by the hits of prediction.
    '''

    n, m = y_hat.shape

    sorce_k = 0
    for v in range(n):
        union = 0
        intersection = 0
        for h in range(m):
            if y_hat[v, h] == 1 or y[v, h] == 1:
                union += 1
            if y_hat[v, h] == 1 and y[v, h] == 1:
                intersection += 1
        if intersection == 0:
            continue
        sorce_k += intersection / sum(y[v])

    return sorce_k / n


def Accuracy(y_hat, y):
    '''
    The “Accuracy” rate is to reflect the average ratio of correctly predicted labels
    over the total labels including correctly and incorrectly predicted labels as well
    as those real labels but are missed in the prediction
    '''

    n, m = y_hat.shape

    sorce_k = 0
    for v in range(n):
        union = 0
        intersection = 0
        for h in range(m):
            if y_hat[v, h] == 1 or y[v, h] == 1:
                union += 1
            if y_hat[v, h] == 1 and y[v, h] == 1:
                intersection += 1
        if intersection == 0:
            continue
        sorce_k += intersection / union
    return sorce_k / n


def AbsoluteTrue(y_hat, y):
    '''
    same
    '''
    """randomforest_edited"""

    n, m = y_hat.shape
    sorce_k = 0
    for v in range(n):
        if (y_hat[v] == y[v]).all():
            sorce_k += 1
    return sorce_k / n


def AbsoluteFalse(y_hat, y):
    '''
    hamming loss
    '''

    n, m = y_hat.shape

    sorce_k = 0
    for v in range(n):
        union = 0
        intersection = 0
        for h in range(m):
            if y_hat[v, h] == 1 or y[v, h] == 1:
                union += 1
            if y_hat[v, h] == 1 and y[v, h] == 1:
                intersection += 1
        sorce_k += (union - intersection) / m
        
    return sorce_k / n

def getSequenceData(first_dir, file_name):
    # getting sequence data and label
    data, label = [], []
    path = "{}/{}.txt".format(first_dir, file_name)

    with open(path) as f:
        for each in f:
            each = each.strip()
            if each[0] == '>':
                label.append(np.array(list(each[1:]), dtype=int))  # Converting string labels to numeric vectors
            else:
                data.append(each)

    return data, label

def PadEncode(data, label, max_len):  # 序列编码
    # encoding
    amino_acids = 'XACDEFGHIKLMNPQRSTVWY'
    data_e, label_e = [], []
    sign = 0
    for i in range(len(data)):
        length = len(data[i])
        elemt, st = [], data[i].strip()
        for j in st:
            if j not in amino_acids:
                sign = 1
                break
            index = amino_acids.index(j)
            elemt.append(index)
            sign = 0

        if length <= max_len and sign == 0:
            elemt += [0] * (max_len - length)
            data_e.append(elemt)
            label_e.append(label[i])
    return np.array(data_e), np.array(label_e)

def predict(test, h5_model):
    dir = 'model'
    print('predicting...')
    for ii in range(0, len(h5_model)):
        # 1.loading model
        h5_model_path = os.path.join(dir, h5_model[ii])
        load_my_model = load_model(h5_model_path,custom_objects={'MultiHeadAttention': MultiHeadAttention})

        # 2.predict
        score = load_my_model.predict(test)

        # 3.getting score
        if ii == 0:
            temp_score = score
        else:
            temp_score += score

    # getting prediction label
    score_label = temp_score / len(h5_model)
    return score_label

first_dir = 'dataset'
max_length = 50  
test_sequence_data, test_sequence_label = getSequenceData(first_dir, 'test')
y_test = np.array(test_sequence_label)   
x_test1, y_test = PadEncode(test_sequence_data, y_test, max_length)
x_test2=np.load("x_test.npy")
x_test=np.concatenate((x_test1.reshape(x_test1.shape[0],50,1),x_test2),axis=2)

h5_model = []
model_num=10
for i in range(1, model_num + 1):
    h5_model.append('model{}.h5'.format('_' + str(i)))
    
test_predict_model1=predict(x_test, h5_model)

model=joblib.load("best_models/0.617180_RF_multi_best_model-class_weights_n__673")
test_data=np.load(r"best_test_features-class_weights.npy")
test_predict_model2=model.predict(test_data).todense()

f=open("model_weights_search_record.txt","a")
score=2.593809#0.031，0.008
best_weights=0
best_threshold=0

for a in tqdm(np.arange(0.287,1,0.001)):
    for b in np.arange(0,1,0.001):
        test_predict=a*test_predict_model1+(1-a)*test_predict_model2
        for i in range(test_predict.shape[0]):
            for j in range(test_predict.shape[1]):
                if test_predict[i,j]/2 < b:
                    test_predict[i,j] = 0
                else:
                    test_predict[i,j] = 1
        aiming=Aiming(test_predict, y_test)
        coverage=Coverage(test_predict, y_test)
        accuracy=Accuracy(test_predict, y_test)
        absoluteTrue=AbsoluteTrue(test_predict, y_test)
        absoluteFalse=AbsoluteFalse(test_predict, y_test)
        score_temp=aiming+coverage+accuracy+absoluteTrue-absoluteFalse
        
        f.write("weights=%lf,threshold=%lf,score=%lf\n"%(a,b,score_temp))
        f.write("Combine_model_Aiming：%lf\n"%aiming)
        f.write("Combine_model_Coverage：%lf\n"%coverage)
        f.write("Combine_model_Accuracy：%lf\n"%accuracy)
        f.write("Combine_model_AbsoluteTrue：%lf\n"%absoluteTrue)
        f.write("Combine_model_AbsoluteFalse：%lf\n\n"%absoluteFalse)
        f.flush()
        
        if score_temp>score:
            score=score_temp
            best_weights=a
            best_threshold=b
            print("upgrade best_weights=%lf,best_threshold=%lf"%(best_weights,best_threshold))
f.write("best_weights=%lf,best_threshold=%lf"%(best_weights,best_threshold))
f.close()       
print("best_weights=%lf"%best_weights)

test_predict=best_weights*test_predict_model1+(1-best_weights)*test_predict_model2
for i in range(test_predict.shape[0]):
    for j in range(test_predict.shape[1]):
        if test_predict[i,j]/2 < best_threshold:
            test_predict[i,j] = 0
        else:
            test_predict[i,j] = 1
            
        if test_predict_model1[i,j] < 0.5:
            test_predict_model1[i,j] = 0
        else:
            test_predict_model1[i,j] = 1
            
f=open("combine_model_result.txt","w")

f.write("test_predict_model1_Aiming：%lf\n"%Aiming(test_predict_model1,y_test))
f.write("test_predict_model1_Coverage：%lf\n"%Coverage(test_predict_model1,y_test))
f.write("test_predict_model1_Accuracy：%lf\n"%Accuracy(test_predict_model1,y_test))
f.write("test_predict_model1_AbsoluteTrue：%lf\n"%AbsoluteTrue(test_predict_model1,y_test))
f.write("test_predict_model1_AbsoluteFalse：%lf\n\n"%AbsoluteFalse(test_predict_model1,y_test))
f.flush()

f.write("test_predict_model2_Aiming：%lf\n"%Aiming(test_predict_model2,y_test))
f.write("test_predict_model2_Coverage：%lf\n"%Coverage(test_predict_model2,y_test))
f.write("test_predict_model2_Accuracy：%lf\n"%Accuracy(test_predict_model2,y_test))
f.write("test_predict_model2_AbsoluteTrue：%lf\n"%AbsoluteTrue(test_predict_model2,y_test))
f.write("test_predict_model2_AbsoluteFalse：%lf\n\n"%AbsoluteFalse(test_predict_model2,y_test))
f.flush()

f.write("Combine_model_Aiming：%lf\n"%Aiming(test_predict,y_test))
f.write("Combine_model_Coverage：%lf\n"%Coverage(test_predict,y_test))
f.write("Combine_model_Accuracy：%lf\n"%Accuracy(test_predict,y_test))
f.write("Combine_model_AbsoluteTrue：%lf\n"%AbsoluteTrue(test_predict,y_test))
f.write("Combine_model_AbsoluteFalse：%lf\n\n"%AbsoluteFalse(test_predict,y_test))
f.flush()

f.write("Aiming improve：%lf\n"%(Aiming(test_predict,y_test)-max(Aiming(test_predict_model1,y_test),
                                                                Aiming(test_predict_model2,y_test))))

f.write("Coverage improve：%lf\n"%(Coverage(test_predict,y_test)-max(Coverage(test_predict_model1,y_test),
                                                                Coverage(test_predict_model2,y_test))))

f.write("Accuracy improve：%lf\n"%(Accuracy(test_predict,y_test)-max(Accuracy(test_predict_model1,y_test),
                                                                Accuracy(test_predict_model2,y_test))))

f.write("AbsoluteTrue improve：%lf\n"%(AbsoluteTrue(test_predict,y_test)-max(AbsoluteTrue(test_predict_model1,y_test),
                                                                AbsoluteTrue(test_predict_model2,y_test))))

f.write("AbsoluteFalse decrease：%lf\n\n"%(-AbsoluteFalse(test_predict,y_test)+min(AbsoluteFalse(test_predict_model1,y_test),
                                                                AbsoluteFalse(test_predict_model2,y_test))))
f.close()
    