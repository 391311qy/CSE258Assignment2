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

