# test
from __future__ import print_function
from pyimagesearch.ir import BagOfVisualWords
import numpy as np

np.random.seed(84)
vocab = np.random.uniform(size=(3, 36))
features = np.random.uniform(size=(100, 36))
bovw = BagOfVisualWords(vocab, sparse=False)
hist = bovw.describe(features)
print("[INFO] BOVW histogram: {}".format(hist))

print("---------------------------------------------------")
print("---------------------------------------------------")
print("---------------------------------------------------")
np.random.seed(42)
vocab = np.random.uniform(size=(5, 36))
features = np.random.uniform(size=(500, 36))
bovw = BagOfVisualWords(vocab, sparse=False)
hist = bovw.describe(features)
print("[INFO] BOVW histogram: {}".format(hist))