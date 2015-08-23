#coding:utf-8
import sys
import codecs
import MeCab

argvs = sys.argv
fname = argvs[1]

f = codecs.open(fname,"r","utf8")
lines = f.readlines()
f.close()

m = MeCab.Tagger("-Ochasen")


for line in lines:
    sentence = ""
    words = line.split() 
    label = words.pop(0) #極性を取り出す
    
    for word in words:
        sentence += word
    
    print label, sentence
    print m.parse(sentence.encode("utf8"))

