#multinomial model
#NB classifier
#alpha = 2

from collections import defaultdict

#---------- start training -----------

trainfile = open('train_kakaku.txt','r') #train file

num_positivesentence = 0
num_negativesentence = 0
wordspositive = {}  #dictionary for positive
wordsnegative = {}  #dictionary for negative 
words = {} #dictionary for all

C = 2       #number of classes

for str in trainfile:
  str = str.rstrip('\n')
  array = str.split(' ')

  #get a class of the sentence
  if array.pop(0) == "+1": 
    num_positivesentence+=1  #count positive sentences 
    for word in array:
      wordspositive[word] = wordspositive.get(word,0) + 1
    

  else:
    num_negativesentence+=1  #count negative senteces
    for word in array:
      wordsnegative[word] = wordsnegative.get(word,0) + 1
    
  for word in array:
    words[word] = words.get(word,0) + 1

num_positivetokens = 0
num_negativetokens = 0
num_differentwords = len(words)

for v in wordspositive.values():
  num_positivetokens+=v #count the number of tokens in positive class
for v in wordsnegative.values():
  num_negativetokens+=v #count the number of tokens in negative class

qw_positive = {}
qw_negative = {}


for k,v in wordspositive.items(): #calculate q-param in positive class
  qw = float(v+1)/float(num_positivetokens + num_differentwords)
  qw_positive[k] = qw

for k,v in wordsnegative.items(): #calculate q-param in negative class
  qw = float(v+1)/float(num_negativetokens + num_differentwords)
  qw_negative[k] = qw

numsentence = num_positivesentence + num_negativesentence #number of input sentences
probpositive_train = float(num_positivesentence + 1)/float(numsentence + C)
probnegative_train = float(num_negativesentence + 1)/float(numsentence + C)

#---------- finish training ----------


#---------- start testing ------------

testfile = open('test_kakaku.txt','r') #test file
num_inputsentence=0
num_right=0
for str in testfile:
  str = str.rstrip('\n')
  array = str.split(' ')

  num_inputsentence+=1
  #get the right class of the input sentence
  rightclass = array.pop(0) 
 
  #classify  
  probpositive_test = probpositive_train
  probnegative_test = probnegative_train
  
  for word in array:
    
    q_positive = qw_positive.get(word,0)
    q_negative = qw_negative.get(word,0)

    if q_positive != 0 or q_negative !=0:
      probpositive_test *= q_positive
      probnegative_test *= q_negative

  if probpositive_test > probnegative_test:
    if rightclass == '+1':
      num_right+=1
  else:
    if rightclass == '-1':
      num_right+=1

print num_inputsentence
print float(num_right)/float(num_inputsentence)

#---------- finish testing -----------


print "Pp:%f" % probpositive_train
print "Pn:%f" % probnegative_train
print "Number of positive sentence:%d" % num_positivesentence
print "Number of negative sentence:%d" % num_negativesentence
print "Number of input sentences:%d" % numsentence

print "Number of different positive-words:%d" % len(wordspositive)
print "Number of different negative-words:%d" % len(wordsnegative)
print "Number of different words:%d" % num_differentwords 

print "Number of tokens in positive class:%d" % num_positivetokens
print "Number of tokens in negative class:%d" % num_negativetokens
