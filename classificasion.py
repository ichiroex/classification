import codecs
from gensim import corpora

f = codecs.open("word_list.txt", "r", "utf8")
lines = f.readlines()
f.close()

word_list = []
for line in lines:
    word_list.append(line.split())

dic = corpora.Dictionary(word_list)
for k,v in sorted(dic.items(), key=lambda x:x[0]):
    print k,v 

