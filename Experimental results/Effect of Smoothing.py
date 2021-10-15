#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.bleu_score import SmoothingFunction


# In[ ]:


import csv
import json

preds=[]
refs=[]
auth_1 =[]
auth_2 = []
auth_3 =[]
with open('human_annotations.csv') as csvfile:
    ader = csv.reader(csvfile)
    for row in ader:
        refs.append(row[1])
        preds.append(row[0])
        auth_1.append(row[2])
        auth_2.append(row[3])
        auth_3.append(row[4])
        
refs_n = refs[:100]
preds_n = preds[:100]

auth_1 = auth_1[:100]
for i in range(0, len(auth_1)):
    auth_1[i] = int(auth_1[i])
#print(auth_1)

auth_2 = auth_2[:100]
for i in range(0, len(auth_2)):
    auth_2[i] = int(auth_2[i])
#print(auth_2)

auth_3 = auth_3[:100]
for i in range(0, len(auth_3)):
    auth_3[i] = int(auth_3[i])
#print(auth_3)


# In[ ]:


#normalised author 1 scores
norm_auth_1 =[]
max1 = max(auth_1)
min1 = min(auth_1)

for a in range(len(auth_1)):
    norm_auth_1.append((auth_1[a])/(max1))
print(norm_auth_1)


# In[ ]:


#normalised author 2 scores
norm_auth_2 =[]
max1 = max(auth_2)
min1 = min(auth_2)

for a in range(len(auth_2)):
    norm_auth_2.append((auth_2[a])/(max1))
print(norm_auth_2)


# In[ ]:


#normalised author 3 scores
norm_auth_3 =[]
max1 = max(auth_3)
min1 = min(auth_3)

for a in range(len(auth_3)):
    norm_auth_3.append((auth_3[a])/(max1))
print(norm_auth_3)


# In[ ]:


#Average normalised author scores
avg_norm_score = []
for i in range(len(auth_1)):
    avg_norm_score.append(round((norm_auth_1[i]+norm_auth_2[i]+norm_auth_3[i])/3,2))
print(avg_norm_score)


# # BLEU4 without smoothing

# In[ ]:


bleu4=[]
for i in range(len(auth_1)):
    bleu4.append(round(sentence_bleu([refs[i]], preds[i]),2))
print(bleu4)


# In[ ]:


norm_bleu4 = []

max_b4=max(bleu4)
min_b4=min(bleu4)

for a in range(len(bleu4)):
    norm_bleu4.append(round(((bleu4[a])/(max_b4)),2))
print(norm_bleu4)


# In[ ]:


from scipy.stats import spearmanr
# calculate spearman's correlation
coef, p = spearmanr(norm_bleu4, avg_norm_score)
print('Spearmans correlation coefficient: %.3f' % coef)
# interpret the significance
alpha = 0.05
if p > alpha:
	print('uncorrelated (fail to reject H0) p=%.3f' % p)
else:
	print('correlated (reject H0) p=%.3f' % p)


# # BLEU4 with smoothing

# BLEUNorm 

# In[ ]:


bleun=[]
for i in range(len(auth_1)):
    bleun.append(round(sentence_bleu([refs[i]], preds[i],smoothing_function=SmoothingFunction().method2),2))
print(bleun)


# In[ ]:


norm_bleun = []

max_bn=max(bleun)
min_bn=min(bleun)

for a in range(len(bleun)):
    norm_bleun.append(round(((bleun[a])/(max_bn)),2))
print(norm_bleun)


# In[ ]:


from scipy.stats import spearmanr
# calculate spearman's correlation
coef, p = spearmanr(norm_bleun, avg_norm_score)
print('Spearmans correlation coefficient: %.3f' % coef)
# interpret the significance
alpha = 0.05
if p > alpha:
	print('uncorrelated (fail to reject H0) p=%.3f' % p)
else:
	print('correlated (reject H0) p=%.3f' % p)


# BLEUCC

# In[ ]:


bleucc=[]
for i in range(len(auth_1)):
    bleucc.append(round(sentence_bleu([refs[i]], preds[i],smoothing_function=SmoothingFunction().method5),2))
print(bleucc)


# In[ ]:


norm_bleucc = []

max_bcc=max(bleucc)
min_bcc=min(bleucc)

for a in range(len(bleucc)):
    norm_bleucc.append(round(((bleucc[a])/(max_bcc)),2))
print(norm_bleucc)


# In[ ]:


from scipy.stats import spearmanr
# calculate spearman's correlation
coef, p = spearmanr(norm_bleucc, avg_norm_score)
print('Spearmans correlation coefficient: %.3f' % coef)
# interpret the significance
alpha = 0.05
if p > alpha:
	print('uncorrelated (fail to reject H0) p=%.3f' % p)
else:
	print('correlated (reject H0) p=%.3f' % p)

