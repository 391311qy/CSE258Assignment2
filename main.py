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

