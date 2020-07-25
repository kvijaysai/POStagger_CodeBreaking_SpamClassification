#!/usr/local/bin/python3
# CSCI B551 Fall 2019
#
# Authors: vivband-akmokka-vikond
#
# based on skeleton code by D. Crandall, 11/2019
#
# ./break_code.py : attack encryption
#


import random
import math
import copy 
import sys
import encode
import collections
import itertools
import numpy as np
import operator
import time
from heapq import nlargest


def cal_probs(corpus):
    """Given corpus this function gives, first character probabilities and 
    probability of next character given previous character for all english alphabet combinations"""
    
    corpus_words = corpus.split(' ')    
    corpus_words = list(filter(None, corpus_words))
    #calculate first character probabilities(P(W^0))                                                                             
    first_char_prob = collections.Counter(map(lambda x: x[0], corpus_words))
    total = sum(first_char_prob.values(), )
    first_char_prob = {k: v / total for k, v in first_char_prob.items()}
    
    #calculate probability of next character given previous character (P(W^j+1|W^j))
    alphabets = list(map(chr,list(range(ord('a'), ord('z')+1))))
    alphabet_tuples = list(itertools.product(alphabets, alphabets))
    consecutive_char_counts = dict.fromkeys(alphabet_tuples,0)
    
    for word in corpus_words:
        for i in range(len(word)-1):
            consecutive_char_counts[(word[i],word[i+1])] += 1
    char_cnt = collections.Counter(itertools.chain.from_iterable(corpus_words))
    total_char_cnt = sum(char_cnt.values(), )
    for key, val in consecutive_char_counts.items():
        consecutive_char_counts[key] = (val+1)/total_char_cnt    
    
    return first_char_prob, consecutive_char_counts

def cal_doc_prob(string, frst_char_prob, consec_char_prob):
    """Given string and probabilities from corpus, this function caluates document probability, i.e., string here"""
    string_words = string.split(' ')
    string_words = list(filter(None, string_words))
    pd = 0
    for word in string_words:
        pd += math.log(frst_char_prob[word[0]])
        for i in range(len(word)-1):
            pd += math.log(consec_char_prob[(word[i],word[i+1])])
    return pd

 
def guess_encrypt(inp_replace, inp_rearrange):
    """This function guesses either of the decryption tables by slightly changing inp_replace or inp_rearrange"""
    rearrange_table = copy.deepcopy(inp_rearrange)
    replace_table = copy.deepcopy(inp_replace)
    if np.random.uniform() > 0.5:  
        letter_swap = []
        while len(set(letter_swap)) < 2:
            letter_swap = list(map(chr, [random.randint(ord('a'), ord('z')) for _ in range(2)]))
        replace_table[letter_swap[0]], replace_table[letter_swap[1]] = replace_table[letter_swap[1]], replace_table[letter_swap[0]]
    else:
        i1, i2 = random.sample(range(4), 2)
        rearrange_table[i1], rearrange_table[i2] = rearrange_table[i2], rearrange_table[i1]

    return replace_table, rearrange_table
    
def first_guess():
    """This function gives the initial guess of both the tables"""
    letters=list(range(ord('a'), ord('z')+1))
    random.shuffle(letters)
    replace_table = dict(zip(map(chr, range(ord('a'), ord('z')+1)), map(chr, letters)))
    rearrange_table = list(range(0,4))
    random.shuffle(rearrange_table)
    return replace_table, rearrange_table

# put your code here!
def break_code(string, corpus):
    """This is the main break code function which decrypts the encrypted file"""
    first_char_prob, consecutive_char_prob = cal_probs(corpus)
    rep1, rear1 = first_guess()
    T = encode.encode(string, rep1, rear1)
    T_doc_prob = cal_doc_prob(T, first_char_prob, consecutive_char_prob)
    
    time_end = time.time() + 60 * 9 # this tells number of minutes to run
    
    population = []
    loop_count = 0
    while time.time() < time_end:
        rep2, rear2 = guess_encrypt(rep1, rear1)
        That = encode.encode(string, rep2, rear2)
        That_doc_prob = cal_doc_prob(That, first_char_prob, consecutive_char_prob)
        if That_doc_prob > T_doc_prob:
            rep1 = copy.deepcopy(rep2)
            rear1 = copy.deepcopy(rear2)
            T_doc_prob = copy.deepcopy(That_doc_prob)
        elif np.random.binomial(n=1, p=math.exp(That_doc_prob - T_doc_prob)):
            rep1 = copy.deepcopy(rep2)
            rear1 = copy.deepcopy(rear2)
            T_doc_prob = copy.deepcopy(That_doc_prob)
    
        ## Append best string to population    
        if T_doc_prob > That_doc_prob:
            population.append((encode.encode(string, rep1, rear1), T_doc_prob))
        else:
            population.append((encode.encode(string, rep2, rear2), That_doc_prob))
        
        ## After every 100 loops store only top 100 strings with their population based on high probability
        if loop_count%100 == 0:
           population = nlargest(100, population, key=operator.itemgetter(1)) 
        loop_count += 1
        
    return max(population, key=operator.itemgetter(1))[0]
    


if __name__== "__main__":
    start_time = time.time()

    if(len(sys.argv) != 4):
        raise Exception("usage: ./break_code.py coded-file corpus output-file")

    encoded = encode.read_clean_file(sys.argv[1])
    corpus = encode.read_clean_file(sys.argv[2])
    decoded = break_code(encoded, corpus)
    print("--- %s minutes ---" % ((time.time() - start_time)/60))
    
    with open(sys.argv[3], "w") as file:
        print(decoded, file=file)
    

