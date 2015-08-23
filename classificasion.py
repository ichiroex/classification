#coding: utf-8
import codecs
from gensim import corpora, matutils
from sklearn.ensemble import RandomForestClassifier

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
    
        #さらに特徴ベクトルに変換
        dense = list(matutils.corpus2dense([vec], num_terms=len(dic)).T[0])
        data_test.append(dense) #訓練データリストに追加
    
    print label_test
    
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
    
    #学習
    estimator = RandomForestClassifier()
    estimator.fit(data_train, label_train)
    
    data_test, label_test = make_test_data(dic)

    #予測
    label_predict = estimator.predict(data_test)
    cnt = 0
    for i in range(len(label_predict)):
        print label_predict[i], label_test[i]
        
        if label_predict[i] == label_test[i]:
            cnt += 1

    val = float(cnt) / float(len(label_predict))
    print float(val)
    

