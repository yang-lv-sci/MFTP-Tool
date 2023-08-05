# -*- coding: utf-8 -*-
"""
Created on Wed Jan  4 20:17:15 2023

@author: lvyang
"""
import numpy as np
from skmultilearn.problem_transform import BinaryRelevance
from skmultilearn.problem_transform import ClassifierChain
from skmultilearn.problem_transform import LabelPowerset
#from skmultilearn.adapt import MLkNN
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import make_scorer
import joblib
from tqdm import tqdm

def tanimoto_coefficient(p_vec, q_vec):
    """
    This method implements the cosine tanimoto coefficient metric
    :param p_vec: vector one
    :param q_vec: vector two
    :return: the tanimoto coefficient between vector one and two
    """
    pq = np.dot(p_vec, q_vec)
    p_square = np.linalg.norm(p_vec)
    q_square = np.linalg.norm(q_vec)
    return pq / (p_square + q_square - pq)

def multi_MDMR(train_features,train_label,test_features):
    import numpy as np
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn import metrics
    from sklearn.metrics import auc
    #step1:特征命名（如果有需要可能会用这个进行存储）
    feature_name=[str(i) for i in range(len(train_features))]
    
    #step2：MRMD得分对feature_array进行逆排序，得分越高的特征重要性越高，排的列数越靠前,
    RVi=[]
    DVi=[]
    MRMD_scorei=[]
    last_index=14601
    for i in tqdm(range(last_index,18201)):#feature_array.shape[1]表示统计数组的列数
        DV=0.0
        RV=0
        for q in range(train_label.shape[1]):
            RV+=np.corrcoef(train_features[:,i], train_label[:,q])#计算x中每一列与y的皮尔逊相关系数
        RVi.append(RV[0,1])
        for j in range(0,train_features.shape[1]):
            EDij=np.linalg.norm(train_features[:,i]-train_features[:,j])#范数计算使用欧式距离公式
            CDij=np.dot(train_features[:,i],train_features[:,j])/(np.linalg.norm(train_features[:,i])*(np.linalg.norm(train_features[:,j])))#余弦距离
            TCij=tanimoto_coefficient(train_features[:,i], train_features[:,j])
            DV+=EDij
            DV+=CDij
            DV-=TCij
        DVi.append(DV/train_features.shape[1])#计算x中每一列到x中所有列的欧氏距离，然后再求平均值
        MRMD_scorei.append(RVi[i-last_index]+ DVi[i-last_index])#得到MRMD得分
        if i%10==0 or i==train_features.shape[1]-1:
            np.save("MDMR_score(%d)"%(last_index), arr=MRMD_scorei)
        
    """根据x中各个特征提取指标的MRMD得分，由大到小对各个指标进行排序
    即MRMD越大的指标,该列的数据越具有代表性，应优先让随机森林数据选择器使用"""
    
    MRMD_scorei = np.array(MRMD_scorei,dtype=np.float32)#转为数组
    rank=np.argsort(MRMD_scorei)#numpy.argsort() 函数返回的是数组值从小到大的索引值。
    rank=rank[::-1]  #rank逆序
    train_features_rank=train_features[...,rank] #例feature_array[...,1]表示取第一列元素
    
    #feature_array_copy=feature_array_copy[...,rank]
    
#    rank_list=list(rank)
    """对特征名字也进行排序与train_features_rank对应"""
#    feature_name_rank=[feature_name[x] for x in rank_list]
    
    #step3：用随机森林剔除冗余数据，寻找最高的AUC指标的特征组合
    MDMR_train_data,MDMR_test_data,MDMR_train_label,MDMR_test_label=train_test_split(train_features_rank,
                                                                                     train_label,test_size=0.2,
                                                                                     random_state=0,stratify=train_label)
    
    # 运行随机森林，寻找最高的AbsoluteTrue指标的特征组合，即 max_key的值
    conclusion = {}#创建检索字典
    
    for cnt in range(train_features_rank.shape[1]):
        
        X_train = MDMR_train_data[..., 0:cnt+1]#取前cnt项
        X_test = MDMR_test_data[..., 0:cnt+1]#取前cnt项
        
        clf = LabelPowerset(RandomForestClassifier(random_state=0,n_jobs=-1,verbose=0))
        clf = clf.fit(X_train,MDMR_train_label)
    
        predict_y_test = clf.predict(X_test)
        
        conclusion[cnt] = AbsoluteTrue(predict_y_test,MDMR_test_label)
        print('共%d次随机森林计算，已完成%s次' %(train_features_rank.shape[1],cnt))
    
    # 找到conclusion字典中最大值对应的键
    for key, value in conclusion.items():
        if value == max(conclusion.values()):
            max_key = key
    
    best_train_features=train_features_rank[:,0:max_key]
    best_test_features=test_features[...,rank][:,0:max_key]
    
    return best_train_features,best_test_features

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

# 用一个基于的分类器
# 初始化二元关联多标签分类器 
"""classifier = BinaryRelevance(RandomForestClassifier())

# 训练
classifier.fit(X_train, y_train)

# 预测
predictions = classifier.predict(X_test)
print("二元ACC：",Accuracy(predictions, y_test))

# 使用分类器链

# 用一个基于高斯朴素贝叶斯的分类器
# 初始化分类器链
classifier = ClassifierChain(RandomForestClassifier())
 
# 训练
classifier.fit(X_train, y_train)

# 预测
predictions = classifier.predict(X_test)
print("分类器链ACC：",Accuracy(predictions, y_test))

# 使用LP法

# 用一个基于高斯朴素贝叶斯的分类器"""

def main():
    X_train=np.load("x_train.npy").reshape(7872,50*531)
    y_train=np.load("y_train.npy")
    X_test=np.load("x_test.npy").reshape(1969,50*531)
    y_test=np.load("y_test.npy")
    X_train,X_test=multi_MDMR(X_train,y_train,X_test)
    np.save(flie="best_RF_train_feature",arr=X_train)
    np.save(flie="best_RF_test_feature",arr=X_test)
    
    # 初始化LP多标签分类器
    #classifier = LabelPowerset(RandomForestClassifier(n_jobs=-1))
    
    parameters = {'classifier__n_estimators':range(100,1000,10)}
    
    my_scoring = make_scorer(Accuracy, greater_is_better=True)
    clf = GridSearchCV(LabelPowerset(RandomForestClassifier(n_jobs=-1,random_state=22,verbose=1)),
                       param_grid=parameters, scoring=my_scoring,cv=5)
    clf.fit(X_train, y_train)
    print(clf.best_params_, clf.best_score_)
    
    # 训练
    classifier=clf.best_estimator_
    classifier.fit(X_train, y_train)
    joblib.dump(classifier, 'RF_multi_model')
    # 预测
    predictions = classifier.predict(X_test)
    
    print("LP法Aiming：",Aiming(predictions, y_test))
    print("LP法Coverage：",Coverage(predictions, y_test))
    print("LP法Accuracy：",Accuracy(predictions, y_test))
    print("LP法AbsoluteTrue：",AbsoluteTrue(predictions, y_test))
    print("LP法AbsoluteFalse：",AbsoluteFalse(predictions, y_test))

if __name__ == '__main__':
    main()
