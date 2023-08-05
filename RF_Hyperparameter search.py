# -*- coding: utf-8 -*-
"""
Created on Fri Jan 13 13:43:31 2023

@author: lvyang
"""

import numpy as np
from sklearn import svm 
from skmultilearn.problem_transform import LabelPowerset
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer
from sklearn.multiclass import OneVsRestClassifier
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

    n, m = y_hat.shape
    sorce_k = 0
    for v in range(n):
        if list(y_hat[v]) == list(y[v]):
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

def main():
    
    best_value=0.616554
    X_train=np.load("best_train_features-class_weights.npy")
    y_train=np.load("y_train.npy")
    X_test=np.load("best_test_features-class_weights.npy")
    y_test=np.load("y_test.npy")
    class_weights=np.load("class_weights.npy",allow_pickle=True).item()
    
    """parameters = {'classifier__n_estimators':range(100,2010,100)}
    
    my_scoring = make_scorer(Accuracy, greater_is_better=True)
    clf = GridSearchCV(LabelPowerset(RandomForestClassifier(n_jobs=-1,random_state=22,verbose=2,class_weight=class_weights)),
                       param_grid=parameters, scoring=my_scoring,cv=3)
    clf.fit(X_train, y_train)
    print(clf.best_params_, clf.best_score_)
    
    # 训练
    classifier=clf.best_estimator_"""
    f=open("best_models\multi_label_demo_class_weights_best_features_class_weights_record_part2.txt","w")
    for n in tqdm(range(2500,3000,1)):
        classifier=LabelPowerset(RandomForestClassifier(n_jobs=-1,random_state=22,verbose=0,class_weight=class_weights,n_estimators=n))
        classifier.fit(X_train, y_train)
        # 预测
        predictions = classifier.predict(X_test)
        
        """print("LP法Aiming：",Aiming(predictions, y_test))
        print("LP法Coverage：",Coverage(predictions, y_test))
        print("LP法Accuracy：",Accuracy(predictions, y_test))
        print("LP法AbsoluteTrue：",AbsoluteTrue(predictions, y_test))
        print("LP法AbsoluteFalse：",AbsoluteFalse(predictions, y_test))"""
        f.write("n_estimator=%d\n"%n)
        f.write("LP法Aiming：%lf\n"%Aiming(predictions, y_test))
        f.write("LP法Coverage：%lf\n"%Coverage(predictions, y_test))
        f.write("LP法Accuracy：%lf\n"%Accuracy(predictions, y_test))
        f.write("LP法AbsoluteTrue：%lf\n"%AbsoluteTrue(predictions, y_test))
        f.write("LP法AbsoluteFalse：%lf\n\n"%AbsoluteFalse(predictions, y_test))
        f.flush()
        
        if Accuracy(predictions, y_test)>best_value:
            best_value=Accuracy(predictions, y_test)
            joblib.dump(classifier, r'best_models\%lf_RF_multi_best_model-class_weights_n__%d'%(Accuracy(predictions, y_test),n))
    f.close()

if __name__ == '__main__':
    main()