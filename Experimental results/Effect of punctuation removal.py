#!/usr/bin/env python
# coding: utf-8

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


#Reference and predicted without punctuation
punc = '''!()-[]{};:'"\,<>./?@#$%^&*_~'''
 
#refs_new = []
for i in range(len(auth_1)):
    for ele in refs_n[i]:
        if ele in punc:
            refs_n[i] = refs_n[i].replace(ele, "")

#preds_new = []
for i in range(len(auth_1)):
    for ele in preds_n[i]:
        if ele in punc:
            preds_n[i] = preds_n[i].replace(ele, "")
print(refs_n)
print(preds_n)


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


# In[ ]:


import numpy as np
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.bleu_score import SmoothingFunction
from nltk.translate.meteor_score import meteor_score


# # BLEU4 with punctuation removal

# In[ ]:


bleu4_pr=[]
for i in range(len(auth_1)):
    bleu4_pr.append(round(sentence_bleu([refs_n[i]], preds_n[i]),2))
print(bleu4_pr)


# In[ ]:


norm_bleu4_pr = []

max_b4=max(bleu4_pr)
min_b4=min(bleu4_pr)

for a in range(len(bleu4_pr)):
    norm_bleu4_pr.append(round(((bleu4_pr[a])/(max_b4)),2))
print(norm_bleu4_pr)


# In[ ]:


from scipy.stats import spearmanr
# calculate spearman's correlation
coef, p = spearmanr(norm_bleu4_pr, avg_norm_score)
print('Spearmans correlation coefficient: %.3f' % coef)
# interpret the significance
alpha = 0.05
if p > alpha:
	print('uncorrelated (fail to reject H0) p=%.3f' % p)
else:
	print('correlated (reject H0) p=%.3f' % p)


# # BLEU4 without punctuation removal 

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


# # BLEUNorm with punctuation removal

# In[ ]:


bleun_pr=[]
for i in range(len(auth_1)):
    bleun_pr.append(round(sentence_bleu([refs_n[i]], preds_n[i],smoothing_function=SmoothingFunction().method2),2))
print(bleun_pr)


# In[ ]:


norm_bleun_pr = []

max_bn=max(bleun_pr)
min_bn=min(bleun_pr)

for a in range(len(bleun_pr)):
    norm_bleun_pr.append(round(((bleun_pr[a])/(max_bn)),2))
print(norm_bleun_pr)


# In[ ]:


from scipy.stats import spearmanr
# calculate spearman's correlation
coef, p = spearmanr(norm_bleun_pr, avg_norm_score)
print('Spearmans correlation coefficient: %.3f' % coef)
# interpret the significance
alpha = 0.05
if p > alpha:
	print('uncorrelated (fail to reject H0) p=%.3f' % p)
else:
	print('correlated (reject H0) p=%.3f' % p)


# # BLEUNorm without punctuation removal

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


# # BLEUCC with punctuation removal

# In[ ]:


bleucc_pr=[]
for i in range(len(auth_1)):
    bleucc_pr.append(round(sentence_bleu([refs_n[i]], preds_n[i],smoothing_function=SmoothingFunction().method5),2))
print(bleucc_pr)


# In[ ]:


norm_bleucc_pr = []

max_bcc=max(bleucc_pr)
min_bcc=min(bleucc_pr)

for a in range(len(bleucc_pr)):
    norm_bleucc_pr.append(round(((bleucc_pr[a])/(max_bcc)),2))
print(norm_bleucc_pr)


# In[ ]:


from scipy.stats import spearmanr
# calculate spearman's correlation
coef, p = spearmanr(norm_bleucc_pr, avg_norm_score)
print('Spearmans correlation coefficient: %.3f' % coef)
# interpret the significance
alpha = 0.05
if p > alpha:
	print('uncorrelated (fail to reject H0) p=%.3f' % p)
else:
	print('correlated (reject H0) p=%.3f' % p)


# # BLEUCC without punctuation removal

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


# In[ ]:


get_ipython().system('pip install rouge')
from rouge import Rouge 
rouge = Rouge()

def rouge1_scores(reference, hypothesis):
    rouge = Rouge()
    scores = rouge.get_scores(hypothesis, reference)
    dict=scores[0]
    dict.values()
    temp0 =list(dict.values())[0]
    temp1= list(dict.values())[1]
    temp2= list(dict.values())[2]
    ROGUE_1 = temp0['f']
    #ROGUE_2 = temp1['f']
    #ROGUE_L = temp2['f']
    return ROGUE_1

def rouge2_scores(reference, hypothesis):
    rouge = Rouge()
    scores = rouge.get_scores(hypothesis, reference)
    dict=scores[0]
    dict.values()
    temp0 =list(dict.values())[0]
    temp1= list(dict.values())[1]
    temp2= list(dict.values())[2]
    #ROGUE_1 = temp0['f']
    ROGUE_2 = temp1['f']
    #ROGUE_L = temp2['f']
    return ROGUE_2

def rougeL_scores(reference, hypothesis):
    rouge = Rouge()
    scores = rouge.get_scores(hypothesis, reference)
    dict=scores[0]
    dict.values()
    temp0 =list(dict.values())[0]
    temp1= list(dict.values())[1]
    temp2= list(dict.values())[2]
    #ROGUE_1 = temp0['f']
    #ROGUE_2 = temp1['f']
    ROGUE_L = temp2['f']
    return ROGUE_L


# # ROUGE-1 with punctuation removal

# In[ ]:


rouge1_pr=[]
for i in range(len(auth_1)):
    rouge1_pr.append(round(rouge1_scores(preds_n[i],refs_n[i]),2))
print(rouge1_pr)


# In[ ]:


norm_r1_pr = []

max_r1=max(rouge1_pr)
min_r1=min(rouge1_pr)

for a in range(len(rouge1_pr)):
    norm_r1_pr.append(round(((rouge1_pr[a])/(max_r1)),2))
print(norm_r1_pr)


# In[ ]:


from scipy.stats import spearmanr
# calculate spearman's correlation
coef, p = spearmanr(norm_r1_pr, avg_norm_score)
print('Spearmans correlation coefficient: %.3f' % coef)
# interpret the significance
alpha = 0.05
if p > alpha:
	print('uncorrelated (fail to reject H0) p=%.3f' % p)
else:
	print('correlated (reject H0) p=%.3f' % p)


# # ROUGE-1 without punctuation removal

# In[ ]:


rouge1=[]
for i in range(len(auth_1)):
    rouge1.append(round(rouge1_scores(preds[i],refs[i]),2))
print(rouge1)


# In[ ]:


norm_r1 = []

max_r1=max(rouge1)
min_r1=min(rouge1)

for a in range(len(rouge1)):
    norm_r1.append(round(((rouge1[a])/(max_r1)),2))
print(norm_r1)


# In[ ]:


from scipy.stats import spearmanr
# calculate spearman's correlation
coef, p = spearmanr(norm_r1, avg_norm_score)
print('Spearmans correlation coefficient: %.3f' % coef)
# interpret the significance
alpha = 0.05
if p > alpha:
	print('uncorrelated (fail to reject H0) p=%.3f' % p)
else:
	print('correlated (reject H0) p=%.3f' % p)


# # ROUGE-2 with punctuation removal

# In[ ]:


rouge2_pr=[]
for i in range(len(auth_1)):
    rouge2_pr.append(round(rouge2_scores(preds_n[i],refs_n[i]),2))
print(rouge2_pr)


# In[ ]:


norm_r2_pr = []

max_r2=max(rouge2_pr)
min_r2=min(rouge2_pr)

for a in range(len(rouge2_pr)):
    norm_r2_pr.append(round(((rouge2_pr[a])/(max_r2)),2))
print(norm_r2_pr)


# In[ ]:


from scipy.stats import spearmanr
# calculate spearman's correlation
coef, p = spearmanr(norm_r2_pr, avg_norm_score)
print('Spearmans correlation coefficient: %.3f' % coef)
# interpret the significance
alpha = 0.05
if p > alpha:
	print('uncorrelated (fail to reject H0) p=%.3f' % p)
else:
	print('correlated (reject H0) p=%.3f' % p)


# # ROUGE-2 without punctuation removal

# In[ ]:


rouge2=[]
for i in range(len(auth_1)):
    rouge2.append(round(rouge2_scores(preds[i],refs[i]),2))
print(rouge2)


# In[ ]:


norm_r2 = []

max_r2=max(rouge2)
min_r2=min(rouge2)

for a in range(len(rouge2)):
    norm_r2.append(round(((rouge2[a])/(max_r2)),2))
print(norm_r2)


# In[ ]:


from scipy.stats import spearmanr
# calculate spearman's correlation
coef, p = spearmanr(norm_r2, avg_norm_score)
print('Spearmans correlation coefficient: %.3f' % coef)
# interpret the significance
alpha = 0.05
if p > alpha:
	print('uncorrelated (fail to reject H0) p=%.3f' % p)
else:
	print('correlated (reject H0) p=%.3f' % p)


# # ROUGE-L with punctuation removal

# In[ ]:


rougel_pr=[]
for i in range(len(auth_1)):
    rougel_pr.append(round(rougeL_scores(preds_n[i],refs_n[i]),2))
print(rougel_pr)


# In[ ]:


norm_rl_pr = []

max_rl=max(rougel_pr)
min_rl=min(rougel_pr)

for a in range(len(rougel_pr)):
    norm_rl_pr.append(round(((rougel_pr[a])/(max_rl)),2))
print(norm_rl_pr)


# In[ ]:


from scipy.stats import spearmanr
# calculate spearman's correlation
coef, p = spearmanr(norm_rl_pr, avg_norm_score)
print('Spearmans correlation coefficient: %.3f' % coef)
# interpret the significance
alpha = 0.05
if p > alpha:
	print('uncorrelated (fail to reject H0) p=%.3f' % p)
else:
	print('correlated (reject H0) p=%.3f' % p)


# # ROUGE-L without punctuation removal

# In[ ]:


rougel=[]
for i in range(len(auth_1)):
    rougel.append(round(rougeL_scores(preds_n[i],refs_n[i]),2))
print(rougel)


# In[ ]:


norm_rl = []

max_rl=max(rougel)
min_rl=min(rougel)

for a in range(len(rougel)):
    norm_rl.append(round(((rougel[a])/(max_rl)),2))
print(norm_rl)


# In[ ]:


from scipy.stats import spearmanr
# calculate spearman's correlation
coef, p = spearmanr(norm_rl, avg_norm_score)
print('Spearmans correlation coefficient: %.3f' % coef)
# interpret the significance
alpha = 0.05
if p > alpha:
	print('uncorrelated (fail to reject H0) p=%.3f' % p)
else:
	print('correlated (reject H0) p=%.3f' % p)


# In[ ]:


import numpy as np


# In[ ]:


def wer_score(hyp, ref, print_matrix=False):
    N = len(hyp)
    M = len(ref)
    L = np.zeros((N,M))
    for i in range(0, N):
        for j in range(0, M):
            if min(i,j) == 0:
                L[i,j] = max(i,j)
            else:
                deletion = L[i-1,j] + 1
                insertion = L[i,j-1] + 1
                sub = 1 if hyp[i] != ref[j] else 0
                substitution = L[i-1,j-1] + sub
                L[i,j] = min(deletion, min(insertion, substitution))
                #print("{} - {}: del {} ins {} sub {} s {}".format(hyp[i], ref[j], deletion, insertion, substitution, sub))
    if print_matrix:
        print("WER matrix ({}x{}): ".format(N, M))
        print(L)
    return int(L[N-1, M-1])


# In[ ]:


def ter(hyp, ref):
    ref = str(ref).split()
    r=len(ref)
    hyp=str(hyp).split()
    wer=wer_score(hyp, ref, print_matrix = False)
    return wer/r


# # TER without punctuation removal

# In[ ]:


ter_=[]
for i in range(len(auth_1)):
    ter_.append(round(ter(preds[i],refs[i]),2))
print(ter_)


# In[ ]:


norm_t = []

max_t=max(ter_)
min_t=min(ter_)

for a in range(len(ter_)):
    norm_t.append(round(((ter_[a])/(max_t)),2))
print(norm_t)


# In[ ]:


t_sim = []
for i in range(len(norm_t)):
    t_sim.append(round((1-norm_t[i]),2))
print(t_sim)


# In[ ]:


from scipy.stats import spearmanr
# calculate spearman's correlation
coef, p = spearmanr(t_sim, avg_norm_score)
print('Spearmans correlation coefficient: %.3f' % coef)
# interpret the significance
alpha = 0.05
if p > alpha:
	print('uncorrelated (fail to reject H0) p=%.3f' % p)
else:
	print('correlated (reject H0) p=%.3f' % p)


# # TER with punctuation removal

# In[ ]:


ter_pr=[]
for i in range(len(auth_1)):
    ter_pr.append(round(ter(preds_n[i],refs_n[i]),2))
print(ter_pr)


# In[ ]:


norm_t_pr = []

max_t=max(ter_pr)
min_t=min(ter_pr)

for a in range(len(ter_pr)):
    norm_t_pr.append(round(((ter_pr[a])/(max_t)),2))
print(norm_t_pr)


# In[ ]:


t_sim_pr = []
for i in range(len(norm_t_pr)):
    t_sim_pr.append(round((1-norm_t_pr[i]),2))
print(t_sim_pr)


# In[ ]:


from scipy.stats import spearmanr
# calculate spearman's correlation
coef, p = spearmanr(t_sim_pr, avg_norm_score)
print('Spearmans correlation coefficient: %.3f' % coef)
# interpret the significance
alpha = 0.05
if p > alpha:
	print('uncorrelated (fail to reject H0) p=%.3f' % p)
else:
	print('correlated (reject H0) p=%.3f' % p)


# # METEOR without punctuation removal

# In[ ]:


from nltk.translate.meteor_score import meteor_score


# In[ ]:


meteor=[]
for i in range(len(auth_1)):
    meteor.append(round(meteor_score([refs[i]], preds[i]),2))
print(meteor)


# In[ ]:


norm_m = []

max_m=max(meteor)
min_m=min(meteor)

for a in range(len(meteor)):
    norm_m.append(round(((meteor[a])/(max_m)),2))
print(norm_m)


# In[ ]:


from scipy.stats import spearmanr
# calculate spearman's correlation
coef, p = spearmanr(norm_m, avg_norm_score)
print('Spearmans correlation coefficient: %.3f' % coef)
# interpret the significance
alpha = 0.05
if p > alpha:
	print('uncorrelated (fail to reject H0) p=%.3f' % p)
else:
	print('correlated (reject H0) p=%.3f' % p)


# # METEOR with punctuation removal

# In[ ]:


meteor_pr=[]
for i in range(len(auth_1)):
    meteor_pr.append(round(meteor_score([refs_n[i]], preds_n[i]),2))
print(meteor_pr)


# In[ ]:


norm_m_pr = []

max_m=max(meteor_pr)
min_m=min(meteor_pr)

for a in range(len(meteor_pr)):
    norm_m_pr.append(round(((meteor_pr[a])/(max_m)),2))
print(norm_m_pr)


# In[ ]:


from scipy.stats import spearmanr
# calculate spearman's correlation
coef, p = spearmanr(norm_m_pr, avg_norm_score)
print('Spearmans correlation coefficient: %.3f' % coef)
# interpret the significance
alpha = 0.05
if p > alpha:
	print('uncorrelated (fail to reject H0) p=%.3f' % p)
else:
	print('correlated (reject H0) p=%.3f' % p)


# # METEOR-NEXT without punctuation removal

# In[ ]:


from nltk.stem.porter import PorterStemmer
from nltk.corpus import wordnet
from itertools import chain, product


def _generate_enums(hypothesis, reference, preprocess=str.lower):
    """
    Takes in string inputs for hypothesis and reference and returns
    enumerated word lists for each of them

    :param hypothesis: hypothesis string
    :type hypothesis: str
    :param reference: reference string
    :type reference: str
    :preprocess: preprocessing method (default str.lower)
    :type preprocess: method
    :return: enumerated words list
    :rtype: list of 2D tuples, list of 2D tuples
    """
    hypothesis_list = list(enumerate(preprocess(hypothesis).split()))
    reference_list = list(enumerate(preprocess(reference).split()))
    return hypothesis_list, reference_list


def exact_match(hypothesis, reference):
    """
    matches exact words in hypothesis and reference
    and returns a word mapping based on the enumerated
    word id between hypothesis and reference

    :param hypothesis: hypothesis string
    :type hypothesis: str
    :param reference: reference string
    :type reference: str
    :return: enumerated matched tuples, enumerated unmatched hypothesis tuples,
             enumerated unmatched reference tuples
    :rtype: list of 2D tuples, list of 2D tuples,  list of 2D tuples
    """
    hypothesis_list, reference_list = _generate_enums(hypothesis, reference)
    return _match_enums(hypothesis_list, reference_list)



def _match_enums(enum_hypothesis_list, enum_reference_list):
    """
    matches exact words in hypothesis and reference and returns
    a word mapping between enum_hypothesis_list and enum_reference_list
    based on the enumerated word id.

    :param enum_hypothesis_list: enumerated hypothesis list
    :type enum_hypothesis_list: list of tuples
    :param enum_reference_list: enumerated reference list
    :type enum_reference_list: list of 2D tuples
    :return: enumerated matched tuples, enumerated unmatched hypothesis tuples,
             enumerated unmatched reference tuples
    :rtype: list of 2D tuples, list of 2D tuples,  list of 2D tuples
    """
    word_match = []
    for i in range(len(enum_hypothesis_list))[::-1]:
        for j in range(len(enum_reference_list))[::-1]:
            if enum_hypothesis_list[i][1] == enum_reference_list[j][1]:
                word_match.append(
                    (enum_hypothesis_list[i][0], enum_reference_list[j][0])
                )
                (enum_hypothesis_list.pop(i)[1], enum_reference_list.pop(j)[1])
                break
    return word_match, enum_hypothesis_list, enum_reference_list


def _enum_stem_match(
    enum_hypothesis_list, enum_reference_list, stemmer=PorterStemmer()
):
    """
    Stems each word and matches them in hypothesis and reference
    and returns a word mapping between enum_hypothesis_list and
    enum_reference_list based on the enumerated word id. The function also
    returns a enumerated list of unmatched words for hypothesis and reference.

    :param enum_hypothesis_list:
    :type enum_hypothesis_list:
    :param enum_reference_list:
    :type enum_reference_list:
    :param stemmer: nltk.stem.api.StemmerI object (default PorterStemmer())
    :type stemmer: nltk.stem.api.StemmerI or any class that implements a stem method
    :return: enumerated matched tuples, enumerated unmatched hypothesis tuples,
             enumerated unmatched reference tuples
    :rtype: list of 2D tuples, list of 2D tuples,  list of 2D tuples
    """
    stemmed_enum_list1 = [
        (word_pair[0], stemmer.stem(word_pair[1])) for word_pair in enum_hypothesis_list
    ]

    stemmed_enum_list2 = [
        (word_pair[0], stemmer.stem(word_pair[1])) for word_pair in enum_reference_list
    ]

    word_match, enum_unmat_hypo_list, enum_unmat_ref_list = _match_enums(
        stemmed_enum_list1, stemmed_enum_list2
    )

    enum_unmat_hypo_list = (
        list(zip(*enum_unmat_hypo_list)) if len(enum_unmat_hypo_list) > 0 else []
    )

    enum_unmat_ref_list = (
        list(zip(*enum_unmat_ref_list)) if len(enum_unmat_ref_list) > 0 else []
    )

    enum_hypothesis_list = list(
        filter(lambda x: x[0] not in enum_unmat_hypo_list, enum_hypothesis_list)
    )

    enum_reference_list = list(
        filter(lambda x: x[0] not in enum_unmat_ref_list, enum_reference_list)
    )

    return word_match, enum_hypothesis_list, enum_reference_list


def stem_match(hypothesis, reference, stemmer=PorterStemmer()):
    """
    Stems each word and matches them in hypothesis and reference
    and returns a word mapping between hypothesis and reference

    :param hypothesis:
    :type hypothesis:
    :param reference:
    :type reference:
    :param stemmer: nltk.stem.api.StemmerI object (default PorterStemmer())
    :type stemmer: nltk.stem.api.StemmerI or any class that
                   implements a stem method
    :return: enumerated matched tuples, enumerated unmatched hypothesis tuples,
             enumerated unmatched reference tuples
    :rtype: list of 2D tuples, list of 2D tuples,  list of 2D tuples
    """
    enum_hypothesis_list, enum_reference_list = _generate_enums(hypothesis, reference)
    return _enum_stem_match(enum_hypothesis_list, enum_reference_list, stemmer=stemmer)



def _enum_wordnetsyn_match(enum_hypothesis_list, enum_reference_list, wordnet=wordnet):
    """
    Matches each word in reference to a word in hypothesis
    if any synonym of a hypothesis word is the exact match
    to the reference word.

    :param enum_hypothesis_list: enumerated hypothesis list
    :param enum_reference_list: enumerated reference list
    :param wordnet: a wordnet corpus reader object (default nltk.corpus.wordnet)
    :type wordnet: WordNetCorpusReader
    :return: list of matched tuples, unmatched hypothesis list, unmatched reference list
    :rtype:  list of tuples, list of tuples, list of tuples

    """
    word_match = []
    for i in range(len(enum_hypothesis_list))[::-1]:
        hypothesis_syns = set(
            chain.from_iterable(
                (
                    lemma.name()
                    for lemma in synset.lemmas()
                    if lemma.name().find("_") < 0
                )
                for synset in wordnet.synsets(enum_hypothesis_list[i][1])
            )
        ).union({enum_hypothesis_list[i][1]})
        for j in range(len(enum_reference_list))[::-1]:
            if enum_reference_list[j][1] in hypothesis_syns:
                word_match.append(
                    (enum_hypothesis_list[i][0], enum_reference_list[j][0])
                )
                enum_hypothesis_list.pop(i), enum_reference_list.pop(j)
                break
    return word_match, enum_hypothesis_list, enum_reference_list


def wordnetsyn_match(hypothesis, reference, wordnet=wordnet):
    """
    Matches each word in reference to a word in hypothesis if any synonym
    of a hypothesis word is the exact match to the reference word.

    :param hypothesis: hypothesis string
    :param reference: reference string
    :param wordnet: a wordnet corpus reader object (default nltk.corpus.wordnet)
    :type wordnet: WordNetCorpusReader
    :return: list of mapped tuples
    :rtype: list of tuples
    """
    enum_hypothesis_list, enum_reference_list = _generate_enums(hypothesis, reference)
    return _enum_wordnetsyn_match(
        enum_hypothesis_list, enum_reference_list, wordnet=wordnet
    )



def _enum_allign_words(
    enum_hypothesis_list, enum_reference_list, stemmer=PorterStemmer(), wordnet=wordnet
):
    """
    Aligns/matches words in the hypothesis to reference by sequentially
    applying exact match, stemmed match and wordnet based synonym match.
    in case there are multiple matches the match which has the least number
    of crossing is chosen. Takes enumerated list as input instead of
    string input

    :param enum_hypothesis_list: enumerated hypothesis list
    :param enum_reference_list: enumerated reference list
    :param stemmer: nltk.stem.api.StemmerI object (default PorterStemmer())
    :type stemmer: nltk.stem.api.StemmerI or any class that implements a stem method
    :param wordnet: a wordnet corpus reader object (default nltk.corpus.wordnet)
    :type wordnet: WordNetCorpusReader
    :return: sorted list of matched tuples, unmatched hypothesis list,
             unmatched reference list
    :rtype: list of tuples, list of tuples, list of tuples
    """
    exact_matches, enum_hypothesis_list, enum_reference_list = _match_enums(
        enum_hypothesis_list, enum_reference_list
    )

    stem_matches, enum_hypothesis_list, enum_reference_list = _enum_stem_match(
        enum_hypothesis_list, enum_reference_list, stemmer=stemmer
    )

    wns_matches, enum_hypothesis_list, enum_reference_list = _enum_wordnetsyn_match(
        enum_hypothesis_list, enum_reference_list, wordnet=wordnet
    )

    return (
        sorted(
            exact_matches + stem_matches + wns_matches, key=lambda wordpair: wordpair[0]
        ),
        enum_hypothesis_list,
        enum_reference_list,
    )


def allign_words(hypothesis, reference, stemmer=PorterStemmer(), wordnet=wordnet):
    """
    Aligns/matches words in the hypothesis to reference by sequentially
    applying exact match, stemmed match and wordnet based synonym match.
    In case there are multiple matches the match which has the least number
    of crossing is chosen.

    :param hypothesis: hypothesis string
    :param reference: reference string
    :param stemmer: nltk.stem.api.StemmerI object (default PorterStemmer())
    :type stemmer: nltk.stem.api.StemmerI or any class that implements a stem method
    :param wordnet: a wordnet corpus reader object (default nltk.corpus.wordnet)
    :type wordnet: WordNetCorpusReader
    :return: sorted list of matched tuples, unmatched hypothesis list, unmatched reference list
    :rtype: list of tuples, list of tuples, list of tuples
    """
    enum_hypothesis_list, enum_reference_list = _generate_enums(hypothesis, reference)
    return _enum_allign_words(
        enum_hypothesis_list, enum_reference_list, stemmer=stemmer, wordnet=wordnet
    )



def _count_chunks(matches):
    """
    Counts the fewest possible number of chunks such that matched unigrams
    of each chunk are adjacent to each other. This is used to caluclate the
    fragmentation part of the metric.

    :param matches: list containing a mapping of matched words (output of allign_words)
    :return: Number of chunks a sentence is divided into post allignment
    :rtype: int
    """
    i = 0
    chunks = 1
    while i < len(matches) - 1:
        if (matches[i + 1][0] == matches[i][0] + 1) and (
            matches[i + 1][1] == matches[i][1] + 1
        ):
            i += 1
            continue
        i += 1
        chunks += 1
    return chunks


def single_meteor_score(
    reference,
    hypothesis,
    preprocess=str.lower,
    stemmer=PorterStemmer(),
    wordnet=wordnet,
    alpha=0.85,
    beta=2.35,
    gamma=0.45,
    w_1=1,
    w_2=0.8,
    w_3=0.6
):
    """
    Calculates METEOR score for single hypothesis and reference as per
    "Meteor: An Automatic Metric for MT Evaluation with HighLevels of
    Correlation with Human Judgments" by Alon Lavie and Abhaya Agarwal,
    in Proceedings of ACL.
    http://www.cs.cmu.edu/~alavie/METEOR/pdf/Lavie-Agarwal-2007-METEOR.pdf


    >>> hypothesis1 = 'It is a guide to action which ensures that the military always obeys the commands of the party'

    >>> reference1 = 'It is a guide to action that ensures that the military will forever heed Party commands'


    >>> round(single_meteor_score(reference1, hypothesis1),4)
    0.7398

        If there is no words match during the alignment the method returns the
        score as 0. We can safely  return a zero instead of raising a
        division by zero error as no match usually implies a bad translation.

    >>> round(meteor_score('this is a cat', 'non matching hypothesis'),4)
    0.0

    :param references: reference sentences
    :type references: list(str)
    :param hypothesis: a hypothesis sentence
    :type hypothesis: str
    :param preprocess: preprocessing function (default str.lower)
    :type preprocess: method
    :param stemmer: nltk.stem.api.StemmerI object (default PorterStemmer())
    :type stemmer: nltk.stem.api.StemmerI or any class that implements a stem method
    :param wordnet: a wordnet corpus reader object (default nltk.corpus.wordnet)
    :type wordnet: WordNetCorpusReader
    :param alpha: parameter for controlling relative weights of precision and recall.
    :type alpha: float
    :param beta: parameter for controlling shape of penalty as a
                 function of as a function of fragmentation.
    :type beta: float
    :param gamma: relative weight assigned to fragmentation penality.
    :type gamma: float
    :return: The sentence-level METEOR score.
    :rtype: float
    """
    enum_hypothesis, enum_reference = _generate_enums(
        hypothesis, reference, preprocess=preprocess
    )
    translation_length = len(enum_hypothesis)
    reference_length = len(enum_reference)
    matches, _, _ = _enum_allign_words(enum_hypothesis, enum_reference, stemmer=stemmer)
    print(matches)
    exact_m,_,_ = exact_match(hypothesis, reference)
    print(exact_m)
    stem_m,_,_ = _enum_stem_match(enum_hypothesis, enum_reference, stemmer=PorterStemmer())
    print(stem_m)
    syn_m,_,_ = _enum_wordnetsyn_match(enum_hypothesis, enum_reference, wordnet=wordnet)
    print(syn_m)
    
    
    
    exact_count = len(exact_m)
    stem_count = len((set(stem_m).difference(exact_m)))
    syn_count = len((set(syn_m).difference(stem_m)))
    print(exact_count)
    matches_count = len(matches)
    print(matches_count)
    try:
        precision = float(w_1*exact_count + w_2*stem_count + w_3*syn_count) / translation_length
        recall = float(w_1*exact_count + w_2*stem_count + w_3*syn_count) / reference_length
        #precision = float(matches_count) / translation_length
        #recall = float(matches_count) / reference_length
        fmean = (precision * recall) / (alpha * precision + (1 - alpha) * recall)
        chunk_count = float(_count_chunks(matches))
        frag_frac = chunk_count / matches_count
    except ZeroDivisionError:
        return 0.0
    penalty = gamma * frag_frac ** beta
    return (1 - penalty) * fmean



def meteorn_score(
    references,
    hypothesis,
    preprocess=str.lower,
    stemmer=PorterStemmer(),
    wordnet=wordnet,
    alpha=0.85,
    beta=2.35,
    gamma=0.45,
    w_1=1,
    w_2=0.8,
    w_3=0.6
):
    """
    Calculates METEOR score for hypothesis with multiple references as
    described in "Meteor: An Automatic Metric for MT Evaluation with
    HighLevels of Correlation with Human Judgments" by Alon Lavie and
    Abhaya Agarwal, in Proceedings of ACL.
    http://www.cs.cmu.edu/~alavie/METEOR/pdf/Lavie-Agarwal-2007-METEOR.pdf


    In case of multiple references the best score is chosen. This method
    iterates over single_meteor_score and picks the best pair among all
    the references for a given hypothesis

    >>> hypothesis1 = 'It is a guide to action which ensures that the military always obeys the commands of the party'
    >>> hypothesis2 = 'It is to insure the troops forever hearing the activity guidebook that party direct'

    >>> reference1 = 'It is a guide to action that ensures that the military will forever heed Party commands'
    >>> reference2 = 'It is the guiding principle which guarantees the military forces always being under the command of the Party'
    >>> reference3 = 'It is the practical guide for the army always to heed the directions of the party'

    >>> round(meteor_score([reference1, reference2, reference3], hypothesis1),4)
    0.7398

        If there is no words match during the alignment the method returns the
        score as 0. We can safely  return a zero instead of raising a
        division by zero error as no match usually implies a bad translation.

    >>> round(meteor_score(['this is a cat'], 'non matching hypothesis'),4)
    0.0

    :param references: reference sentences
    :type references: list(str)
    :param hypothesis: a hypothesis sentence
    :type hypothesis: str
    :param preprocess: preprocessing function (default str.lower)
    :type preprocess: method
    :param stemmer: nltk.stem.api.StemmerI object (default PorterStemmer())
    :type stemmer: nltk.stem.api.StemmerI or any class that implements a stem method
    :param wordnet: a wordnet corpus reader object (default nltk.corpus.wordnet)
    :type wordnet: WordNetCorpusReader
    :param alpha: parameter for controlling relative weights of precision and recall.
    :type alpha: float
    :param beta: parameter for controlling shape of penalty as a function
                 of as a function of fragmentation.
    :type beta: float
    :param gamma: relative weight assigned to fragmentation penality.
    :type gamma: float
    :return: The sentence-level METEOR score.
    :rtype: float
    """
    return max(
        [
            single_meteor_score(
                reference,
                hypothesis,
                preprocess=preprocess,
                stemmer=stemmer,
                wordnet=wordnet,
                alpha=alpha,
                beta=beta,
                gamma=gamma,
            )
            for reference in references
        ]
    )


# In[ ]:


meteorn=[]
for i in range(len(auth_1)):
    meteorn.append(round(meteorn_score([refs[i]],preds[i]),2))
print(meteorn)


# In[ ]:


norm_mn = []

max_mn=max(meteorn)
min_mn=min(meteorn)

for a in range(len(meteorn)):
    norm_mn.append(round(((meteorn[a])/(max_mn)),2))
print(norm_mn)


# In[ ]:


from scipy.stats import spearmanr
# calculate spearman's correlation
coef, p = spearmanr(norm_mn, avg_norm_score)
print('Spearmans correlation coefficient: %.3f' % coef)
# interpret the significance
alpha = 0.05
if p > alpha:
	print('uncorrelated (fail to reject H0) p=%.3f' % p)
else:
	print('correlated (reject H0) p=%.3f' % p)


# # METEOR-NEXT with punctuation removal

# In[ ]:


from nltk.stem.porter import PorterStemmer
from nltk.corpus import wordnet
from itertools import chain, product


def _generate_enums(hypothesis, reference, preprocess=str.lower):
    """
    Takes in string inputs for hypothesis and reference and returns
    enumerated word lists for each of them

    :param hypothesis: hypothesis string
    :type hypothesis: str
    :param reference: reference string
    :type reference: str
    :preprocess: preprocessing method (default str.lower)
    :type preprocess: method
    :return: enumerated words list
    :rtype: list of 2D tuples, list of 2D tuples
    """
    hypothesis_list = list(enumerate(preprocess(hypothesis).split()))
    reference_list = list(enumerate(preprocess(reference).split()))
    return hypothesis_list, reference_list


def exact_match(hypothesis, reference):
    """
    matches exact words in hypothesis and reference
    and returns a word mapping based on the enumerated
    word id between hypothesis and reference

    :param hypothesis: hypothesis string
    :type hypothesis: str
    :param reference: reference string
    :type reference: str
    :return: enumerated matched tuples, enumerated unmatched hypothesis tuples,
             enumerated unmatched reference tuples
    :rtype: list of 2D tuples, list of 2D tuples,  list of 2D tuples
    """
    hypothesis_list, reference_list = _generate_enums(hypothesis, reference)
    return _match_enums(hypothesis_list, reference_list)



def _match_enums(enum_hypothesis_list, enum_reference_list):
    """
    matches exact words in hypothesis and reference and returns
    a word mapping between enum_hypothesis_list and enum_reference_list
    based on the enumerated word id.

    :param enum_hypothesis_list: enumerated hypothesis list
    :type enum_hypothesis_list: list of tuples
    :param enum_reference_list: enumerated reference list
    :type enum_reference_list: list of 2D tuples
    :return: enumerated matched tuples, enumerated unmatched hypothesis tuples,
             enumerated unmatched reference tuples
    :rtype: list of 2D tuples, list of 2D tuples,  list of 2D tuples
    """
    word_match = []
    for i in range(len(enum_hypothesis_list))[::-1]:
        for j in range(len(enum_reference_list))[::-1]:
            if enum_hypothesis_list[i][1] == enum_reference_list[j][1]:
                word_match.append(
                    (enum_hypothesis_list[i][0], enum_reference_list[j][0])
                )
                (enum_hypothesis_list.pop(i)[1], enum_reference_list.pop(j)[1])
                break
    return word_match, enum_hypothesis_list, enum_reference_list


def _enum_stem_match(
    enum_hypothesis_list, enum_reference_list, stemmer=PorterStemmer()
):
    """
    Stems each word and matches them in hypothesis and reference
    and returns a word mapping between enum_hypothesis_list and
    enum_reference_list based on the enumerated word id. The function also
    returns a enumerated list of unmatched words for hypothesis and reference.

    :param enum_hypothesis_list:
    :type enum_hypothesis_list:
    :param enum_reference_list:
    :type enum_reference_list:
    :param stemmer: nltk.stem.api.StemmerI object (default PorterStemmer())
    :type stemmer: nltk.stem.api.StemmerI or any class that implements a stem method
    :return: enumerated matched tuples, enumerated unmatched hypothesis tuples,
             enumerated unmatched reference tuples
    :rtype: list of 2D tuples, list of 2D tuples,  list of 2D tuples
    """
    stemmed_enum_list1 = [
        (word_pair[0], stemmer.stem(word_pair[1])) for word_pair in enum_hypothesis_list
    ]

    stemmed_enum_list2 = [
        (word_pair[0], stemmer.stem(word_pair[1])) for word_pair in enum_reference_list
    ]

    word_match, enum_unmat_hypo_list, enum_unmat_ref_list = _match_enums(
        stemmed_enum_list1, stemmed_enum_list2
    )

    enum_unmat_hypo_list = (
        list(zip(*enum_unmat_hypo_list)) if len(enum_unmat_hypo_list) > 0 else []
    )

    enum_unmat_ref_list = (
        list(zip(*enum_unmat_ref_list)) if len(enum_unmat_ref_list) > 0 else []
    )

    enum_hypothesis_list = list(
        filter(lambda x: x[0] not in enum_unmat_hypo_list, enum_hypothesis_list)
    )

    enum_reference_list = list(
        filter(lambda x: x[0] not in enum_unmat_ref_list, enum_reference_list)
    )

    return word_match, enum_hypothesis_list, enum_reference_list


def stem_match(hypothesis, reference, stemmer=PorterStemmer()):
    """
    Stems each word and matches them in hypothesis and reference
    and returns a word mapping between hypothesis and reference

    :param hypothesis:
    :type hypothesis:
    :param reference:
    :type reference:
    :param stemmer: nltk.stem.api.StemmerI object (default PorterStemmer())
    :type stemmer: nltk.stem.api.StemmerI or any class that
                   implements a stem method
    :return: enumerated matched tuples, enumerated unmatched hypothesis tuples,
             enumerated unmatched reference tuples
    :rtype: list of 2D tuples, list of 2D tuples,  list of 2D tuples
    """
    enum_hypothesis_list, enum_reference_list = _generate_enums(hypothesis, reference)
    return _enum_stem_match(enum_hypothesis_list, enum_reference_list, stemmer=stemmer)



def _enum_wordnetsyn_match(enum_hypothesis_list, enum_reference_list, wordnet=wordnet):
    """
    Matches each word in reference to a word in hypothesis
    if any synonym of a hypothesis word is the exact match
    to the reference word.

    :param enum_hypothesis_list: enumerated hypothesis list
    :param enum_reference_list: enumerated reference list
    :param wordnet: a wordnet corpus reader object (default nltk.corpus.wordnet)
    :type wordnet: WordNetCorpusReader
    :return: list of matched tuples, unmatched hypothesis list, unmatched reference list
    :rtype:  list of tuples, list of tuples, list of tuples

    """
    word_match = []
    for i in range(len(enum_hypothesis_list))[::-1]:
        hypothesis_syns = set(
            chain.from_iterable(
                (
                    lemma.name()
                    for lemma in synset.lemmas()
                    if lemma.name().find("_") < 0
                )
                for synset in wordnet.synsets(enum_hypothesis_list[i][1])
            )
        ).union({enum_hypothesis_list[i][1]})
        for j in range(len(enum_reference_list))[::-1]:
            if enum_reference_list[j][1] in hypothesis_syns:
                word_match.append(
                    (enum_hypothesis_list[i][0], enum_reference_list[j][0])
                )
                enum_hypothesis_list.pop(i), enum_reference_list.pop(j)
                break
    return word_match, enum_hypothesis_list, enum_reference_list


def wordnetsyn_match(hypothesis, reference, wordnet=wordnet):
    """
    Matches each word in reference to a word in hypothesis if any synonym
    of a hypothesis word is the exact match to the reference word.

    :param hypothesis: hypothesis string
    :param reference: reference string
    :param wordnet: a wordnet corpus reader object (default nltk.corpus.wordnet)
    :type wordnet: WordNetCorpusReader
    :return: list of mapped tuples
    :rtype: list of tuples
    """
    enum_hypothesis_list, enum_reference_list = _generate_enums(hypothesis, reference)
    return _enum_wordnetsyn_match(
        enum_hypothesis_list, enum_reference_list, wordnet=wordnet
    )



def _enum_allign_words(
    enum_hypothesis_list, enum_reference_list, stemmer=PorterStemmer(), wordnet=wordnet
):
    """
    Aligns/matches words in the hypothesis to reference by sequentially
    applying exact match, stemmed match and wordnet based synonym match.
    in case there are multiple matches the match which has the least number
    of crossing is chosen. Takes enumerated list as input instead of
    string input

    :param enum_hypothesis_list: enumerated hypothesis list
    :param enum_reference_list: enumerated reference list
    :param stemmer: nltk.stem.api.StemmerI object (default PorterStemmer())
    :type stemmer: nltk.stem.api.StemmerI or any class that implements a stem method
    :param wordnet: a wordnet corpus reader object (default nltk.corpus.wordnet)
    :type wordnet: WordNetCorpusReader
    :return: sorted list of matched tuples, unmatched hypothesis list,
             unmatched reference list
    :rtype: list of tuples, list of tuples, list of tuples
    """
    exact_matches, enum_hypothesis_list, enum_reference_list = _match_enums(
        enum_hypothesis_list, enum_reference_list
    )

    stem_matches, enum_hypothesis_list, enum_reference_list = _enum_stem_match(
        enum_hypothesis_list, enum_reference_list, stemmer=stemmer
    )

    wns_matches, enum_hypothesis_list, enum_reference_list = _enum_wordnetsyn_match(
        enum_hypothesis_list, enum_reference_list, wordnet=wordnet
    )

    return (
        sorted(
            exact_matches + stem_matches + wns_matches, key=lambda wordpair: wordpair[0]
        ),
        enum_hypothesis_list,
        enum_reference_list,
    )


def allign_words(hypothesis, reference, stemmer=PorterStemmer(), wordnet=wordnet):
    """
    Aligns/matches words in the hypothesis to reference by sequentially
    applying exact match, stemmed match and wordnet based synonym match.
    In case there are multiple matches the match which has the least number
    of crossing is chosen.

    :param hypothesis: hypothesis string
    :param reference: reference string
    :param stemmer: nltk.stem.api.StemmerI object (default PorterStemmer())
    :type stemmer: nltk.stem.api.StemmerI or any class that implements a stem method
    :param wordnet: a wordnet corpus reader object (default nltk.corpus.wordnet)
    :type wordnet: WordNetCorpusReader
    :return: sorted list of matched tuples, unmatched hypothesis list, unmatched reference list
    :rtype: list of tuples, list of tuples, list of tuples
    """
    enum_hypothesis_list, enum_reference_list = _generate_enums(hypothesis, reference)
    return _enum_allign_words(
        enum_hypothesis_list, enum_reference_list, stemmer=stemmer, wordnet=wordnet
    )



def _count_chunks(matches):
    """
    Counts the fewest possible number of chunks such that matched unigrams
    of each chunk are adjacent to each other. This is used to caluclate the
    fragmentation part of the metric.

    :param matches: list containing a mapping of matched words (output of allign_words)
    :return: Number of chunks a sentence is divided into post allignment
    :rtype: int
    """
    i = 0
    chunks = 1
    while i < len(matches) - 1:
        if (matches[i + 1][0] == matches[i][0] + 1) and (
            matches[i + 1][1] == matches[i][1] + 1
        ):
            i += 1
            continue
        i += 1
        chunks += 1
    return chunks


def single_meteor_score(
    reference,
    hypothesis,
    preprocess=str.lower,
    stemmer=PorterStemmer(),
    wordnet=wordnet,
    alpha=0.85,
    beta=2.35,
    gamma=0.45,
    w_1=1,
    w_2=0.8,
    w_3=0.6
):
    """
    Calculates METEOR score for single hypothesis and reference as per
    "Meteor: An Automatic Metric for MT Evaluation with HighLevels of
    Correlation with Human Judgments" by Alon Lavie and Abhaya Agarwal,
    in Proceedings of ACL.
    http://www.cs.cmu.edu/~alavie/METEOR/pdf/Lavie-Agarwal-2007-METEOR.pdf


    >>> hypothesis1 = 'It is a guide to action which ensures that the military always obeys the commands of the party'

    >>> reference1 = 'It is a guide to action that ensures that the military will forever heed Party commands'


    >>> round(single_meteor_score(reference1, hypothesis1),4)
    0.7398

        If there is no words match during the alignment the method returns the
        score as 0. We can safely  return a zero instead of raising a
        division by zero error as no match usually implies a bad translation.

    >>> round(meteor_score('this is a cat', 'non matching hypothesis'),4)
    0.0

    :param references: reference sentences
    :type references: list(str)
    :param hypothesis: a hypothesis sentence
    :type hypothesis: str
    :param preprocess: preprocessing function (default str.lower)
    :type preprocess: method
    :param stemmer: nltk.stem.api.StemmerI object (default PorterStemmer())
    :type stemmer: nltk.stem.api.StemmerI or any class that implements a stem method
    :param wordnet: a wordnet corpus reader object (default nltk.corpus.wordnet)
    :type wordnet: WordNetCorpusReader
    :param alpha: parameter for controlling relative weights of precision and recall.
    :type alpha: float
    :param beta: parameter for controlling shape of penalty as a
                 function of as a function of fragmentation.
    :type beta: float
    :param gamma: relative weight assigned to fragmentation penality.
    :type gamma: float
    :return: The sentence-level METEOR score.
    :rtype: float
    """
    enum_hypothesis, enum_reference = _generate_enums(
        hypothesis, reference, preprocess=preprocess
    )
    translation_length = len(enum_hypothesis)
    reference_length = len(enum_reference)
    matches, _, _ = _enum_allign_words(enum_hypothesis, enum_reference, stemmer=stemmer)
    print(matches)
    exact_m,_,_ = exact_match(hypothesis, reference)
    print(exact_m)
    stem_m,_,_ = _enum_stem_match(enum_hypothesis, enum_reference, stemmer=PorterStemmer())
    print(stem_m)
    syn_m,_,_ = _enum_wordnetsyn_match(enum_hypothesis, enum_reference, wordnet=wordnet)
    print(syn_m)
    
    
    
    exact_count = len(exact_m)
    stem_count = len((set(stem_m).difference(exact_m)))
    syn_count = len((set(syn_m).difference(stem_m)))
    print(exact_count)
    matches_count = len(matches)
    print(matches_count)
    try:
        precision = float(w_1*exact_count + w_2*stem_count + w_3*syn_count) / translation_length
        recall = float(w_1*exact_count + w_2*stem_count + w_3*syn_count) / reference_length
        #precision = float(matches_count) / translation_length
        #recall = float(matches_count) / reference_length
        fmean = (precision * recall) / (alpha * precision + (1 - alpha) * recall)
        chunk_count = float(_count_chunks(matches))
        frag_frac = chunk_count / matches_count
    except ZeroDivisionError:
        return 0.0
    penalty = gamma * frag_frac ** beta
    return (1 - penalty) * fmean



def meteorn_score(
    references,
    hypothesis,
    preprocess=str.lower,
    stemmer=PorterStemmer(),
    wordnet=wordnet,
    alpha=0.85,
    beta=2.35,
    gamma=0.45,
    w_1=1,
    w_2=0.8,
    w_3=0.6
):
    """
    Calculates METEOR score for hypothesis with multiple references as
    described in "Meteor: An Automatic Metric for MT Evaluation with
    HighLevels of Correlation with Human Judgments" by Alon Lavie and
    Abhaya Agarwal, in Proceedings of ACL.
    http://www.cs.cmu.edu/~alavie/METEOR/pdf/Lavie-Agarwal-2007-METEOR.pdf


    In case of multiple references the best score is chosen. This method
    iterates over single_meteor_score and picks the best pair among all
    the references for a given hypothesis

    >>> hypothesis1 = 'It is a guide to action which ensures that the military always obeys the commands of the party'
    >>> hypothesis2 = 'It is to insure the troops forever hearing the activity guidebook that party direct'

    >>> reference1 = 'It is a guide to action that ensures that the military will forever heed Party commands'
    >>> reference2 = 'It is the guiding principle which guarantees the military forces always being under the command of the Party'
    >>> reference3 = 'It is the practical guide for the army always to heed the directions of the party'

    >>> round(meteor_score([reference1, reference2, reference3], hypothesis1),4)
    0.7398

        If there is no words match during the alignment the method returns the
        score as 0. We can safely  return a zero instead of raising a
        division by zero error as no match usually implies a bad translation.

    >>> round(meteor_score(['this is a cat'], 'non matching hypothesis'),4)
    0.0

    :param references: reference sentences
    :type references: list(str)
    :param hypothesis: a hypothesis sentence
    :type hypothesis: str
    :param preprocess: preprocessing function (default str.lower)
    :type preprocess: method
    :param stemmer: nltk.stem.api.StemmerI object (default PorterStemmer())
    :type stemmer: nltk.stem.api.StemmerI or any class that implements a stem method
    :param wordnet: a wordnet corpus reader object (default nltk.corpus.wordnet)
    :type wordnet: WordNetCorpusReader
    :param alpha: parameter for controlling relative weights of precision and recall.
    :type alpha: float
    :param beta: parameter for controlling shape of penalty as a function
                 of as a function of fragmentation.
    :type beta: float
    :param gamma: relative weight assigned to fragmentation penality.
    :type gamma: float
    :return: The sentence-level METEOR score.
    :rtype: float
    """
    return max(
        [
            single_meteor_score(
                reference,
                hypothesis,
                preprocess=preprocess,
                stemmer=stemmer,
                wordnet=wordnet,
                alpha=alpha,
                beta=beta,
                gamma=gamma,
            )
            for reference in references
        ]
    )


# In[ ]:


meteorn_pr=[]
for i in range(len(auth_1)):
    meteorn_pr.append(round(meteorn_score([refs_n[i]],preds_n[i]),2))
print(meteorn_pr)


# In[ ]:


norm_mn_pr = []

max_mn=max(meteorn_pr)
min_mn=min(meteorn_pr)

for a in range(len(meteorn_pr)):
    norm_mn_pr.append(round(((meteorn_pr[a])/(max_mn)),2))
print(norm_mn_pr)


# In[ ]:


from scipy.stats import spearmanr
# calculate spearman's correlation
coef, p = spearmanr(norm_mn_pr, avg_norm_score)
print('Spearmans correlation coefficient: %.3f' % coef)
# interpret the significance
alpha = 0.05
if p > alpha:
	print('uncorrelated (fail to reject H0) p=%.3f' % p)
else:
	print('correlated (reject H0) p=%.3f' % p)

