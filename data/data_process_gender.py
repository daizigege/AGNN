import codecs
import pandas as pd
import numpy as np
from scipy.sparse import coo_matrix
import codecs
import torch
import os
import re
import time
# 数据集
dataset='ml_latest'
spt=','
words = set()
movGenders = []
f = codecs.open(dataset+'/movies.csv', 'r', 'utf-8', errors='ignore')  # 以防非法字符
s = f.readline()
while s != "":
    s = re.sub(r"\".*\"", " ", s)  # 去掉名字里面有，
    ss = s.split(",")
    i = int(ss[0])
    tag = ss[2]
    tag = re.sub(r"[^a-zA-Z|0-9]", "", tag).lower()
    ts = tag.split("|")
    for t in ts:
        if t not in words:
            words.add(t)
    line = [i, set()]
    for t in ts:
        line[1].add(t)
    movGenders.append(line)
    s = f.readline()
f.close()
genderNum = len(words)

movNum=len(movGenders)
genderIdx = {word: i for i, word in enumerate(words)}
movDicGender = torch.zeros(movNum, genderNum)
for i, gs in movGenders:
    tid = [genderIdx[g] for g in gs]
    movDicGender[i - 1][tid] = 1
print()
torch.save(movDicGender,dataset+ "/gender.npy")
a=torch.load(dataset+ "/gender.npy")
print('a')
