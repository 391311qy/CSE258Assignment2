from collections import defaultdict
import gzip
import random
import math
import scipy
from scipy import optimize
import numpy
import nltk
from nltk.tokenize import word_tokenize
import string
from nltk.stem.porter import *
from sklearn import linear_model
from scipy import spatial
from random import shuffle
from nltk.corpus import stopwords
import itertools
from sklearn.metrics import mean_squared_error
import statistics

# read data
data = []
requiredEntrySet = {"fit", "bust size", "weight", "rating", "rented for", "review_text", "body type", "review_summary", "category", "height", "size", "age"}
def readJSON(path):
	for l in gzip.open(path, 'rt'):
		try:
			d = eval(l)
			curEntries = d.keys()
			if requiredEntrySet.issubset(curEntries):
				yield d
		except Exception as e:
			pass

for d in readJSON("renttherunway_final_data.json.gz"):
	data.append(d)

data = data[:10000]

# process the data to parse the height and the weight
import re

def parse_height(height):
    pattern = re.compile(r'(\d+)\' (\d+)\"')
    match = pattern.match(height)

    if match:
        feet = int(match.group(1))
        inch = int(match.group(2))

        return feet*12 + inch
    else:
        return -1

def parse_weight(weight):
    pattern = re.compile(r'(\d+)lbs')
    match = pattern.match(weight)
    if match:
        return int(match.group(1))
    else:
        return -1


def parse_data(data):
    for d in data:
        height = parse_height(d['height'])
        d['height'] = height

        weight = parse_weight(d['weight'])
        d['weight'] = weight

parse_data(data)

# extract the min and max for regulization
def extract_max(data):
    maxAge = -1
    maxSize = -1
    maxWeight = -1
    maxHeight = -1

    for d in data:
        if int(d["age"]) > maxAge:
            maxAge = int(d["age"])
        if d["size"] > maxSize:
            maxSize = d["size"]
        if d["weight"] > maxWeight:
            maxWeight = d["weight"]
        if d["height"] > maxHeight:
            maxHeight = d["height"]
    
    return maxAge, maxSize, maxWeight, maxHeight

maxAge, maxSize, maxWeight, maxHeight = extract_max(data)

def extract_min(data):
    minAge = maxAge
    minSize = maxSize
    minWeight = maxWeight
    minHeight = maxHeight

    for d in data:
        if int(d["age"]) < minAge:
            minAge = int(d["age"])
        if d["size"] < minSize:
            minSize = d["size"]
        if d["weight"] < minWeight:
            minWeight = d["weight"]
        if d["height"] < minHeight:
            minHeight = d["height"]
    
    return minAge, minSize, minWeight, minHeight

minAge, minSize, minWeight, minHeight = extract_min(data)

# extract the regulized features
def regulize(d, minD, maxD):
    return (d - minD)/(maxD - minD)

def age(datum, minAge, maxAge):
    return regulize(int(datum["age"]), minAge, maxAge)

def size(datum, minSize, maxSize):
    return regulize(datum["size"], minSize, maxSize)

def weight(datum, minWeight, maxWeight):
    return regulize(datum["weight"], minWeight, maxWeight)

def height(datum, minHeight, maxHeight):
    return regulize(datum["height"], minHeight, maxHeight)




# One hot encoding features
Entry = ["fit", "rented for", "body type", "category", "bust size"]

def extract_onehot(data, Entry):
	Maps = {Entry[i]:defaultdict(int) for i in range(len(Entry))}
	for d in data:
		for f in Entry:
			mp = Maps[f]
			if d[f] not in mp:
				mp[d[f]] = len(mp)
	return Maps

# fit: 3; rented for: 9; body type: 7; category: 68; bust size: 101

Maps = extract_onehot(data, Entry)

# Combined single vector
def feature_onehot_combined(datum):
	featureVectors = [[0]*len(Maps[f]) for f in Entry]
	for i in range(len(Entry)):
		f = Entry[i]
		featureVectors[i][Maps[f][d[f]]] = 1
	singleVector = list(itertools.chain.from_iterable(featureVectors))
	singleVector.append(1)
	return singleVector


# separate features (all 5)
def feature_onehot_separated(datum):
	featureVectors = [[0]*len(Maps[f]) for f in Entry]
	for i in range(len(Entry)):
		f = Entry[i]
		featureVectors[i][Maps[f][d[f]]] = 1
	for v in featureVectors:
		v.append(1)
	return featureVectors

# print(feature_onehot_separated(data[22]))

# fit feature
def featureFit(datum):
	feat = [0] * len(Maps['fit'])
	feat[Maps['fit'][d['fit']]] = 1
	feat.append(1)
	return feat

def featureRentedFor(datum):
	feat = [0] * len(Maps['rented for'])
	feat[Maps['rented for'][d['rented for']]] = 1
	feat.append(1)
	return feat

def featureBodyType(datum):
	feat = [0] * len(Maps['body type'])
	feat[Maps['body type'][d['body type']]] = 1
	feat.append(1)
	return feat

def featureCategory(datum):
	feat = [0] * len(Maps['category'])
	feat[Maps['category'][d['category']]] = 1
	feat.append(1)
	return feat

def featureBustSize(datum):
	feat = [0] * len(Maps['bust size'])
	feat[Maps['bust size'][d['bust size']]] = 1
	feat.append(1)
	return feat

#import nltk
#nltk.download('stopwords')

# unigram and bigram
uniWordCount = defaultdict(int)
biWordCount = defaultdict(int)

uniWordCount_rmPunc = defaultdict(int)
biWordCount_rmPunc = defaultdict(int)

uniWordCount_rmPunc_Stop = defaultdict(int)
biWordCount_rmPunc_Stop = defaultdict(int)

punctuation = set(string.punctuation)
stopwords = set(stopwords.words('english'))
punctuation.add('\n')
stemmer = PorterStemmer()

for d in data:
	text = ''.join([c for c in d['review_summary'].lower()])
	for token in text.split():
		uniWordCount[token] += 1
	text = ''.join([c for c in d['review_summary'].lower() if c not in punctuation ])
	for token in text.split():
		uniWordCount_rmPunc[token] += 1
	text = ''.join([c for c in d['review_summary'].lower() if c not in punctuation and c not in stopwords])
	for token in text.split():
		uniWordCount_rmPunc_Stop[token] += 1

	text = ''.join([c for c in d['review_text'].lower() if c not in punctuation ])
	bigrm = nltk.bigrams(nltk.word_tokenize(text))
	for token in bigrm:
		biWordCount_rmPunc[token] += 1
	text = ''.join([c for c in d['review_text'].lower()])
	bigrm = nltk.bigrams(nltk.word_tokenize(text))
	for token in bigrm:
		biWordCount[token] += 1
	text = ''.join([c for c in d['review_text'].lower() if c not in punctuation and c not in stopwords])
	bigrm = nltk.bigrams(nltk.word_tokenize(text))
	for token in bigrm:
		biWordCount_rmPunc_Stop[token] += 1

top_k_uni = 40
top_k_bi = 40

unicounts = [(uniWordCount[w], w) for w in uniWordCount]
unicounts.sort()
unicounts.reverse()
uniwords = [x[1] for x in unicounts[:top_k_uni]]
uniwordId = dict(zip(uniwords, range(len(uniwords))))
uniwordSet = set(uniwords)

unicounts_rmPunc = [(uniWordCount_rmPunc[w], w) for w in uniWordCount_rmPunc]
unicounts_rmPunc.sort()
unicounts_rmPunc.reverse()
uniwords_rmPunc = [x[1] for x in unicounts_rmPunc[:top_k_uni]]
uniwordId_rmPunc = dict(zip(uniwords_rmPunc, range(len(uniwords_rmPunc))))
uniwordSet_rmPunc = set(uniwords_rmPunc)

uniWordCount_rmPunc_Stop = [(uniWordCount_rmPunc_Stop[w], w) for w in uniWordCount]
uniWordCount_rmPunc_Stop.sort()
uniWordCount_rmPunc_Stop.reverse()
uniwords_rmPunc_Stop = [x[1] for x in uniWordCount_rmPunc_Stop[:top_k_uni]]
uniwordId_rmPunc_Stop = dict(zip(uniwords_rmPunc_Stop, range(len(uniwords_rmPunc_Stop))))
uniwordSet_rmPunc_Stop = set(uniwords_rmPunc_Stop)

bicounts = [(biWordCount[w], w) for w in biWordCount]
bicounts.sort()
bicounts.reverse()
biwords = [x[1] for x in bicounts[:top_k_bi]]
biwordId = dict(zip(biwords, range(len(biwords))))
biwordSet = set(biwords)

bicounts_rmPunc = [(biWordCount_rmPunc[w], w) for w in biWordCount_rmPunc]
bicounts_rmPunc.sort()
bicounts_rmPunc.reverse()
biwords_rmPunc = [x[1] for x in bicounts_rmPunc[:top_k_bi]]
biwordId_rmPunc = dict(zip(biwords_rmPunc, range(len(biwords_rmPunc))))
biwordSet_rmPunc = set(biwords_rmPunc)

bicounts_rmPunc_Stop = [(biWordCount_rmPunc_Stop[w], w) for w in biWordCount_rmPunc_Stop]
bicounts_rmPunc_Stop.sort()
bicounts_rmPunc_Stop.reverse()
biwords_rmPunc_Stop = [x[1] for x in bicounts_rmPunc_Stop[:top_k_bi]]
biwordId_rmPunc_Stop = dict(zip(biwords_rmPunc_Stop, range(len(biwords_rmPunc_Stop))))
biwordSet_rmPunc_Stop = set(biwords_rmPunc_Stop)

def featureSummary(datum, rmPunc, rmPuncStop):
	feat = [0]*len(uniwords)
	if rmPunc:
		r = ''.join([c for c in datum['review_summary'].lower() if c not in punctuation])
	elif rmPuncStop:
		r = ''.join([c for c in datum['review_summary'].lower() if c not in punctuation and c not in stopwords])
	else:
		r = ''.join([c for c in datum['review_summary'].lower()])
	for token in r.split():
		if rmPunc:
			if token in uniwords_rmPunc:
				feat[uniwordId_rmPunc[token]] += 1
		elif rmPuncStop:
			if token in uniwords_rmPunc_Stop:
				feat[uniwordId_rmPunc_Stop[token]] += 1
		else:
			if token in uniwords:
				feat[uniwordId[token]] += 1
	feat.append(1)
	return feat

def featureText(datum, rmPunc, rmPuncStop):
	feat = [0]*len(uniwords)
	if rmPunc:
		r = ''.join([c for c in datum['review_text'].lower() if c not in punctuation])
	elif rmPuncStop:
		r = ''.join([c for c in datum['review_text'].lower() if c not in punctuation and c not in stopwords])
	else:
		r = ''.join([c for c in datum['review_text'].lower()])

	for token in r.split():
		if rmPunc:
			if token in biwords_rmPunc:
				feat[biwordId_rmPunc[token]] += 1
		elif rmPuncStop:
			if token in biwords_rmPunc_Stop:
				feat[biwordId_rmPunc_Stop[token]] += 1
		else:
			if token in uniwords:
				feat[biwordId[token]] += 1
	feat.append(1)
	return feat



clf = linear_model.Ridge(1.0, fit_intercept=False) # MSE + 1.0 l2
clf.fit(X, y)
predictions = clf.predict(X_test)
print("mse of prediction is "+ str(mean_squared_error(y_test ,predictions)))
