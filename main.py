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

Entry = ["fit", "rented for", "body type", "category", "bust size"]

def extract_onehot(data, Entry):
	Maps = [defaultdict(int) for i in range(len(Entry))]
	for d in data:
		for i in range(5):
			k, mp = Entry[i], Maps[i]
			if d[k] not in mp:
				mp[d[k]] = len(mp)
	return Maps

# fit: 3
# rented for: 9
# body type: 7
# category: 68
# bust size: 101

Maps = extract_onehot(data, Entry)
# print(len(Maps[4]))
		
def feature_onehot(datum, Maps, Entry):
	vec = [[0]*len(i) for i in Maps]
	for i in range(len(Maps)):
		vec[i][Maps[i][d[Entry[i]]]] = 1
	
	# # add 1 as intercept for each feature vector
	# for v in vec:
	# 	v.append(1)

	# else if you want all features combined as one
	vec = list(itertools.chain.from_iterable(vec))
	vec.append(1)

	return vec

# vec = feature_onehot(data[22], Maps, Entry)
# print(vec)


X = [feature_onehot(d, Maps, Entry) for d in data]
y = [d["rating"] for d in data]








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

unicounts = [(uniWordCount[w], w) for w in uniWordCount]
unicounts.sort()
unicounts.reverse()
uniwords = [x[1] for x in unicounts[:1000]]
uniwordId = dict(zip(uniwords, range(len(uniwords))))
uniwordSet = set(uniwords)

unicounts_rmPunc = [(uniWordCount_rmPunc[w], w) for w in uniWordCount_rmPunc]
unicounts_rmPunc.sort()
unicounts_rmPunc.reverse()
uniwords_rmPunc = [x[1] for x in unicounts_rmPunc[:1000]]
uniwordId_rmPunc = dict(zip(uniwords_rmPunc, range(len(uniwords_rmPunc))))
uniwordSet_rmPunc = set(uniwords_rmPunc)

bicounts = [(biWordCount[w], w) for w in biWordCount]
bicounts.sort()
bicounts.reverse()
biwords = [x[1] for x in bicounts[:1000]]
biwordId = dict(zip(biwords, range(len(biwords))))
biwordSet = set(biwords)

bicounts_rmPunc = [(biWordCount_rmPunc[w], w) for w in biWordCount_rmPunc]
bicounts_rmPunc.sort()
bicounts_rmPunc.reverse()
biwords_rmPunc = [x[1] for x in bicounts_rmPunc[:1000]]
biwordId_rmPunc = dict(zip(biwords_rmPunc, range(len(biwords_rmPunc))))
biwordSet_rmPunc = set(biwords_rmPunc)

uniWordCount_rmPunc_Stop = [(uniWordCount_rmPunc_Stop[w], w) for w in uniWordCount]
uniWordCount_rmPunc_Stop.sort()
uniWordCount_rmPunc_Stop.reverse()
uniwords_rmPunc_Stop = [x[1] for x in uniWordCount_rmPunc_Stop[:1000]]
uniwordId_rmPunc_Stop = dict(zip(uniwords_rmPunc_Stop, range(len(uniwords_rmPunc_Stop))))
uniwordSet_rmPunc_Stop = set(uniwords_rmPunc_Stop)

bicounts_rmPunc_Stop = [(biWordCount_rmPunc_Stop[w], w) for w in biWordCount_rmPunc_Stop]
bicounts_rmPunc_Stop.sort()
bicounts_rmPunc_Stop.reverse()
biwords_rmPunc_Stop = [x[1] for x in bicounts_rmPunc_Stop[:1000]]
biwordId_rmPunc_Stop = dict(zip(biwords_rmPunc_Stop, range(len(biwords_rmPunc_Stop))))
biwordSet_rmPunc_Stop = set(biwords_rmPunc_Stop)

def featureSummary(datum, rmPunc, rmPuncStop):
	feat = [0]*len(uniwords)
	if rmPunc:
		r = ''.join([c for c in datum['review_text'].lower() if c not in punctuation])
	elif rmPuncStop:
		r = ''.join([c for c in datum['review_text'].lower() if c not in punctuation and c not in stopwords])
	else:
		r = ''.join([c for c in datum['review_text'].lower()])
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


