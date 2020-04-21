# Project: CS 168 SP2020 - Miniproject 2 - Question 3
# Author: Ann Wu
# Date: 04/20/2020

import numpy as np
import matplotlib.pyplot as plt
import sys
import csv
from datetime import datetime

narticles = 1000
max_wordid = 0 # this is 'k' for any vector computation

# FIXME :
# Runtime optimization
# - save the hash values for each article into a dictionary of lists.
#       (https://piazza.com/class/k8gkb66j2i53qe?cid=188)
# - Convert bin to dec
#       (https://piazza.com/class/k8gkb66j2i53qe?cid=190)
# - for each such computation we can (similar to previous parts) exploit the sparsity of the vectors and avoid going through all the 60,000+ dimensions.
#       (https://piazza.com/class/k8gkb66j2i53qe?cid=185)
#
# - save hash buckets to CSV files for all d values and read it back in for queries, to separate them!!


# Separate the data into each article's bag of words, where each word = (x,y) where x = wordID and y = freq
def import_data():
    global max_wordid

    data = [[] for i in range(narticles)] # init list of lists for data
    with open('p2_data/data50.csv', 'rb') as csvfile:
        data50reader = csv.reader(csvfile, delimiter=',')
        for row in data50reader:
            data[int(row[0])-1].append((int(row[1]), int(row[2])))
            if (int(row[1]) > max_wordid):
                max_wordid = int(row[1])

    print "max_wordid =", max_wordid
    #print "data = "
    #for i in range(len(data)):
    #    print "\tarticle =", i
    #    print "\twords =", data[i]
    return data

def expand_vector(V):
    V_expanded = np.zeros((max_wordid, 1))
    for word_pair in V:
        V_expanded[word_pair[0]] = word_pair[1]
    return V_expanded


def hyperplane_hash(d, V):
    # Populate (d x k) matrix from Gaussian distribution
    mu, sigma = 0, 1
    Mi = np.random.normal(mu, sigma, (d, max_wordid))

    # Get (d x 1) vector MiV
    MiV = np.dot(Mi, expand_vector(V))

    # Get sgn(MiV)
    MiV_sgn = np.sign(MiV)
    MiV_sgn[MiV_sgn == -1] = 0 # Replace all elements with value -1 with 0
    MiV_rot = np.rot90(MiV_sgn)[0]

    # Convert the array to a binary string
    MiV_str = ""
    for i in range(d):
        MiV_str += str(int(MiV_rot[i]))

    # Convert binary string to int
    MiV_int = int(MiV_str, 2)

    return MiV_int

# Given a dataset (of n vectors), populate the buckets with their hyperplane hash
def populate_buckets(dataset, buckets, d, l):
    vector_idx = 0
    for V in dataset:
        print "mapping article", vector_idx
        if (vector_idx%25 == 0):
            dateTimeObj = datetime.now()
            print "current time =", dateTimeObj
            for bucket in buckets:
                print bucket

        # for each iter, hash the element and insert the current vector_idx into the hash bucket
        for i in range(l):
            bucket_idx = hyperplane_hash(d, V)
            buckets[bucket_idx].append(vector_idx)
        vector_idx += 1

        # Remove duplicates in each bucket
        for i in range(2**d):
            tmp = list(set(buckets[i]))
            buckets[i] = tmp

    return

def get_best_candidate(candidates, Q, dataset):
    best_score = (0,0)
    first = True
    for c_idx in candidates:

        carr = expand_vector(dataset[c_idx])
        qarr = expand_vector(Q)
        #score = (c_idx, np.linalg.norm(carr-qarr)) # use L2 similarity
        score = (c_idx, (np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y)))) # use Cosine similarity
        if first:
            best_score = score
            first = False
        else:
            if (score[1] < best_score[1]):
                best_score = score

    return best_score[0] # return vector index, not actual vector

def query(dataset, buckets, d, l, Q):
    candidates = []

    # Get all candidates
    for i in range(l):
        bucket_idx = hyperplane_hash(d, Q)
        candidates += buckets[bucket_idx]
    tmp = list(set(candidates))
    candidates = tmp # Remove duplicate candidates

    # Return most similar candidate's idx
    return (len(candidates), get_best_candidate(candidates, Q, dataset))

def query_assess(dataset, buckets, d, l):
    q_idx, err_cnt, tot_cnt, tot_candidates = 0, 0, 0, 0
    for Q in dataset:
        print "querying article", q_idx
        if (q_idx%25 == 0):
            dateTimeObj = datetime.now()
            print "current time =", dateTimeObj
            print "err_cnt =", err_cnt, " ; tot_cnt =", tot_cnt

        (cur_candidates, match_idx) = query(dataset, buckets, d, l, Q)
        if (q_idx != match_idx):
            err_cnt += 1
        q_idx += 1
        tot_cnt += 1
        tot_candidates += cur_candidates

    err_perc = err_cnt/tot_cnt
    num_candidates = tot_candidates/narticles
    return (err_perc, num_candidates)

def experiment(d, l, dataset):
    buckets = [[] for i in range(2**d)] # init list of lists for Classification
    populate_buckets(dataset, buckets, d, l)
    return query_assess(dataset, buckets, d, l)

def main():
    l = 128                     # Number of hash tables
    d_lst = list(range(5,21))   # Log2(# buckets)
    dataset = import_data()     # dataset = list of lists

    # Run the experiment for each d
    err_perc_lst = []
    num_candidates_lst = []
    for cur_d in d_lst:
        dateTimeObj = datetime.now()
        print "current time =", dateTimeObj
        print "cur_d =", cur_d
        (err_perc, num_candidates) = experiment(cur_d, l, dataset)
        err_perc_lst.append(err_perc)
        num_candidates_lst.append(num_candidates)

    # plot err_cnt vs Sq_size
    plt.plot(err_perc_lst, num_candidates_lst)
    plt.show()


main()
