#!/usr/bin/env python3

'''
TASK B
'''

from nltk.tokenize import TweetTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import cross_val_score, cross_val_predict, LeaveOneOut
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.datasets import dump_svmlight_file
from sklearn.decomposition import PCA
from sklearn import metrics
import numpy as np
import logging
import codecs

from sklearn.multiclass import OneVsRestClassifier
from random import randint
import gensim.models
import word2vecReaderUtils as utils
from word2vecReader import *
import json

logging.basicConfig(level=logging.INFO)

def parse_dataset(dataset):
    '''Loads the dataset .txt file with label-tweet on each line and parses the dataset.'''
    y = []
    corpus = []
    with open(dataset, 'r') as data_in:
        for line in data_in:
            if not line.lower().startswith("tweet index"):  # discard first line if it contains metadata
                line = line.rstrip()    # remove trailing whitespace
                if "train" in dataset.lower():
                    label = int(line.split("\t")[1])
                    tweet = line.split("\t")[2]
                    y.append(label)
                else:
                    tweet = line.split("\t")[2]
#                     print(tweet)
                corpus.append(tweet)
    if "train" in dataset.lower():
        return corpus, y
    else:
        return corpus

def sort_coo(m):
    tuples = zip(m.indices, m.data)
    return sorted(tuples, key=lambda x: (x[1], x[0]), reverse=True)


def featurize(corpus):
    '''
    Tokenizes and creates TF-IDF BoW vectors.
    :param corpus: A list of strings each string representing document.
    :return: X: A sparse csr matrix of TFIDF-weigted ngram counts.
    '''
    tokenizer = TweetTokenizer(preserve_case=False, reduce_len=True, strip_handles=True).tokenize
    vectorizer = TfidfVectorizer(strip_accents="unicode", analyzer="word", tokenizer=tokenizer, stop_words="english")
    X = vectorizer.fit_transform(corpus)
    strUnits = vectorizer.get_feature_names()
    print("len features",len(vectorizer.get_feature_names())) # to manually check if the tokens are reasonable

    print(sort_coo(X[0]))
    print(X[0].argmax())
    print(type(X))
    return X

def ngramFeaturize(corpus): # Res with ngram_range=(1, 2), LR: 0.6451
    tokenizer = TweetTokenizer(preserve_case=False, reduce_len=True, strip_handles=True).tokenize
    #vectorizer = TfidfVectorizer(strip_accents="unicode", analyzer="word", tokenizer=tokenizer, stop_words="english")
    bigram_vectorizer = CountVectorizer(ngram_range=(1, 2),tokenizer=tokenizer, min_df=1) #token_pattern=r'\b\w+\b',
    analyze = bigram_vectorizer.build_analyzer()

    X = bigram_vectorizer.fit_transform(corpus).toarray()
    strUnits = bigram_vectorizer.get_feature_names()
    print("len features",len(bigram_vectorizer.get_feature_names()))

    return X
    

def wvVectors(corpus):
    tweetVectors = []
    tokenizer = TweetTokenizer(preserve_case=False, reduce_len=True, strip_handles=True).tokenize
    #wvModel = Word2Vec.load_word2vec_format('/Users/omid/Downloads/word2vec_twitter_model/word2vec_twitter_model.bin', binary=True)
    wvModel = Word2Vec.load_word2vec_format('/Users/shiva/Downloads/word2vec_twitter_model/word2vec_twitter_model.bin', binary=True)
    emojiModel = gensim.models.KeyedVectors.load_word2vec_format('../../extra_resources/emoji2vec.bin', binary=True)
    unknowns = []
    for tweet in corpus:
        t = tokenizer(tweet)
        sentVectors = []
        for word in t:
            #if word.startswith('#'):
            #    word = word[1:]
            if word in wvModel:
                sentVectors.append(wvModel[word])
            elif word in emojiModel:
                # print('emoji:', word)
                sentVectors.append(np.concatenate((emojiModel[word], np.zeros(100))))
            '''
            else:
                vocabNum = len(wvModel.vocab)-1
                wordVocab = list(wvModel.vocab.keys())
                sentVectors.append(wvModel[wordVocab[randint(0,vocabNum)]])
                unknowns.append(word)
            '''
            ### We should think if we want to have a symbolic vector (zero for example) for words that are not present in the w2v
        if len(sentVectors)==0:
            print("empty sentence",t)
        tweetVectors.append(sentVectors)
    print("len(unknownWords)", len(unknowns))
    return tweetVectors

def wvConcatVectorsFeaturize(corpus):
    concatenatedVectors = []
    tVectors = wvVectors(corpus)
    #maxlength_w = max([len(t) for t in tVectors])
    #print(maxlength_w)
    maxlength_w = 41
    for vecs in tVectors:
        concatVec = []
        for i in range(maxlength_w):
            if i<len(vecs):
                concatVec = np.concatenate((concatVec,vecs[i]))
            else:
                concatVec = np.concatenate((concatVec,np.zeros(400)))
        concatenatedVectors.append(concatVec)
    # pca = PCA(n_components=100)
    # pca.fit_transform(concatenatedVectors)
    return concatenatedVectors
                
def wvMeanVectorsFeaturize(corpus):
    meanVectors = []
    tVectors = wvVectors(corpus)
    for tweetVec in tVectors:
        m = np.mean(tweetVec, axis=0)
        v = np.var(tweetVec, axis=0)
        std = np.std(tweetVec, axis=0)
        if len(tweetVec)==0:
            m = np.zeros(400)
            v = np.zeros(400)
            std = np.zeros(400)
        meanVectors.append(m)
    return meanVectors

def wvMeanVectorsKsectionsFeaturize(corpus, k):    #with k==1(2 splits), LR classifier: 0.6742
                                                # same with emojis, slight decrease to 0.6721
    def section(dataVecs):
        foldSize = int(len(dataVecs)/2)
        sec1 = dataVecs[0:foldSize]
        sec2 = dataVecs[foldSize:]
        return sec1, sec2
        
    vectors = []
    tVectors = wvVectors(corpus)
    for tweetVec in tVectors:
        secs = section(tweetVec)
        
        for i in range(1,k):
            temp = ()
            for s in secs:
                temp= temp+section(s)
            secs = temp
            
        ms = ()
        for sec in secs:
            m = np.mean(sec, axis=0)
            if len(sec) == 0:
                m = np.zeros(400)
            ms = ms+(m,)
        
        vectors.append(np.concatenate(ms))
    return vectors
                        
if __name__ == "__main__":
    # Experiment settings

    # Dataset: SemEval2018-T4-train-taskA.txt or SemEval2018-T4-train-taskB.txt
    #DATASET_FP = "../../datasets/train/SemEval2018-T4-train-taskA.txt"
    # DATASET_FP = "../../datasets/train/SemEval2018-T4-train-taskA_emoji.txt"
    DATASET_FP = "../../datasets/train/SemEval2018-T4-train-taskB_emoji.txt"
    TASK = "B" # Define, A or B
    FNAME = './predictions-task' + TASK + '2.txt'
    PREDICTIONSFILE = open(FNAME, "w")
    EXTRA_FEATURES = 1

    K_FOLDS = 10 # 10-fold crossvalidation
    
    #CLF = LinearSVC(random_state=0) # the default, non-parameter optimized linear-kernel SVM
    clf1 = LogisticRegression(multi_class = 'ovr', solver = 'liblinear', class_weight = {0:1, 1:1, 2:3,3:3}) # (C=1e3)
    clf2 = LogisticRegression(multi_class = 'ovr', solver = 'liblinear', class_weight = {0:1, 1:1, 2:1, 3:1})
    clf3 = LogisticRegression(multi_class = 'ovr', solver = 'liblinear', class_weight = {0:1, 1:1, 2:2, 3:2})

    CLF = VotingClassifier(estimators=[('lr1', clf1), ('lr2', clf2), ('lr3', clf3)], voting='soft')

    ###extra_corpus, extra_y = 
    # Loading dataset and featurised simple Tfidf-BoW model
    corpus, y = parse_dataset(DATASET_FP)

    test_corpus  = parse_dataset('../../datasets/test_TaskB/SemEval2018-T3_input_test_taskB_emoji.txt')

    #X = featurize(corpus)          # The reason that their model is very fast even with 11,793 features, is that
                                    # their input is sparse.csr.csr_matrix
    X_bigram = ngramFeaturize(corpus+test_corpus)
    
    X1 = X_bigram[0:len(corpus)]
    #X = wvMeanVectorsFeaturize(corpus)
    #X = wvMeanVectorsKsectionsFeaturize(corpus, 1)      #wvMeanVectorsFeaturize(corpus)
    X = wvConcatVectorsFeaturize(corpus)

    X = list(X)
    y = np.array(y)
    #print("shape X after concatination:", X.shape)

    if EXTRA_FEATURES:
        extraFeatures1 = np.load(open('../../taskA/code/feature_vecs/corenlp_newIntensityFeats.npy', 'rb'))
        extraFeatures2 = np.load(open('../../taskA/code/feature_vecs/contrast_feats_new.npy', 'rb'))
        extraFeatures3 = np.load(open('../../taskA/code/feature_vecs/feats_n_chunks.npy', 'rb')) # read only 0:48 of this
        extraFeatures4 = np.load(open('../../taskA/code/feature_vecs/feats.npy', 'rb')) # read only 0:24
        extraFeatures5 = np.load(open('../../taskA/code/feature_vecs/all_feats.npy', 'rb')) # read only -5:end
        extraFeatures6 = np.load('../../taskB/code/feature_vecs/topic_feats_10_2chunks.npy') # was EFFECTIVE
        extraFeatures7 = np.load('../../taskB/code/feature_vecs/topic_feats_10.npy') 
         
        extraFeatures =[np.concatenate((extraFeatures1[i], [extraFeatures2[i]], extraFeatures3[i][0:48],
                                   extraFeatures4[i][0:24], extraFeatures5[i][-5:],
                                   extraFeatures6[i], extraFeatures7[i] #, extraFeatures8[i]
                                   )) for i in range(len(extraFeatures1))]
 
        #extraFeatures = np.load(open('no_difference_baby.npy', 'rb'))  # weigthed features
        
        for i in range(len(X)):
            X[i] = np.concatenate((X[i],X1[i],extraFeatures[i]))
    print("X dimension",len(X[0]))
    X = np.array(X)
        
    class_counts = np.asarray(np.unique(y, return_counts=True)).T.tolist()
    print ("class counts:",class_counts)

    
    # Returns an array of the same size as 'y' where each entry is a prediction obtained by cross validated
    predicted = cross_val_predict(CLF, X, y, cv=K_FOLDS)

    # Modify F1-score calculation depending on the task
    if TASK.lower() == 'a':
        score = metrics.f1_score(y, predicted, pos_label=1)
    elif TASK.lower() == 'b':
         # if you set average to None, it will return results for each class separately 
        score = metrics.f1_score(y, predicted, average=None)
        score_ = metrics.f1_score(y, predicted, average='macro') 
    print ("F1-score Task", TASK, score)
    print ("F1-score Task", TASK, score_)
    for p in predicted:
        PREDICTIONSFILE.write("{}\n".format(p))
    PREDICTIONSFILE.close()
    
        
    #X_test = ngramFeaturize(test_corpus)
    X1_test = X_bigram[len(corpus):]
    X_test = wvConcatVectorsFeaturize(test_corpus)

    X_test = list(X_test)
    #print("shape X after concatination:", X.shape)

    if EXTRA_FEATURES:
         
        extraFeatures = np.load("feature_vecs/all_test_feats_TaskB.npy")
        
        for i in range(len(X_test)):
            X_test[i] = np.concatenate((X_test[i],X1_test[i], extraFeatures[i]))
    print("X_test dimension",len(X_test[0]))
    X_test = np.array(X_test)

    print("Fit on the whole Train ...")
    CLF.fit(X, y)

    print("Ready to TEST")


    y_test_predicted = CLF.predict(X_test)
    
    with open('test_prediction_TaskB_shiva.txt', 'w') as f:
        for y in y_test_predicted:
            f.write(str(y)+"\n")

