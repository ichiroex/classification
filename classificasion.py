#coding: utf-8
"""
author: ichiroex

ref: http://qiita.com/yasunori/items/31a23eb259482e4824e2
"""

import codecs
from gensim import corpora, matutils
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.grid_search import GridSearchCV
from sklearn import datasets
from sklearn.decomposition import TruncatedSVD

def readfile(fname):
    
    f = codecs.open(fname, "r", "utf8")
    lines = f.readlines()
    f.close()

    word_list = []
    all_words = []
    label_train = []
    for line in lines:
        tmp = line.split()

        label = tmp.pop(0)
        if label == "+1":
            label_train.append(1)
        else:
            label_train.append(0)

        all_words += tmp
        word_list.append(tmp)
    
    return word_list, all_words, label_train

def make_test_data(dic):
    word_list, all_words, label_test = readfile("test_kakaku.txt") #test data
    
    #入力した文の全単語から辞書を作る
    #dic = corpora.Dictionary(word_list)
    
    #訓練データをベクトル化
    data_test = []
    for words in word_list:
        
        #入力文をbag-of-wordsで表現
        vec = dic.doc2bow(words)
    
        #さらに特徴ベクトルに変換. 訓練データから作成した辞書を使う.
        dense = list(matutils.corpus2dense([vec], num_terms=len(dic)).T[0])
        data_test.append(dense) #訓練データリストに追加
    
    return data_test, label_test

if __name__ == "__main__":

    word_list, all_words, label_train = readfile("train_kakaku.txt") #train data
    
    #入力した文の全単語から辞書を作る
    dic = corpora.Dictionary(word_list)
    
    #辞書を表示
    """
    for k,v in sorted(dic.items(), key=lambda x:x[0]):
        print k,v 
    """
    
    #訓練データをベクトル化
    data_train = []
    for words in word_list:
        
        #入力文をbag-of-wordsで表現
        vec = dic.doc2bow(words)
        
        #さらに特徴ベクトルに変換
        dense = list(matutils.corpus2dense([vec], num_terms=len(dic)).T[0])
        data_train.append(dense) #訓練データリストに追加
    
    #特徴量の次元を圧縮
    """
    lsa = TruncatedSVD(500)
    reduced_data_train = lsa.fit_transform(data_train)
    dic = corpora.Dictionary(reduced_data_train)
    """
    #------ 学習(start) ------
    #Random Forest Classifier
    estimator = RandomForestClassifier()
    estimator.fit(data_train, label_train)
    
    #Suppor Vector Machine
    """
    estimator = svm.SVC()
    estimator.fit(data_train, label_train)
    """
    #------ 学習(end) ------
    

    #テストデータをベクトル化
    data_test, label_test = make_test_data(dic)

    #------ 予測 ------
    label_predict = estimator.predict(data_test)
    
    """
    "ris = datasets.load_iris()
    features = iris.data
    labels = iris.target
    print features
    print labels
    """
    
    #正解率を表示
    print estimator.score(data_test, label_test)

    """
    " この掛け合わせを試す
    tuned_parameters = [{'n_estimators': [10, 30, 50, 70, 90, 110, 130, 150], 'max_features': ['auto', 'sqrt', 'log2', None]}]

    clf = GridSearchCV(RandomForestClassifier(), tuned_parameters, cv=2, scoring='accuracy', n_jobs=-1)
    clf.fit(data_train, label_train)

    print("ベストパラメタを表示")
    print(clf.best_estimator_)

    print("トレーニングデータでCVした時の平均スコア")
    for params, mean_score, all_scores in clf.grid_scores_:
        print("{:.3f} (+/- {:.3f}) for {}".format(mean_score, all_scores.std() / 2, params))

    y_true, y_pred = label_test_s, clf.predict(data_test)
    print(classification_report(y_true, y_pred))
    """

