# Project: CS 168 SP2020 - Miniproject 2 - Question 3
# Author: Ann Wu
# Date: 04/20/2020

import numpy as np
import matplotlib.pyplot as plt
import sys
import csv
from datetime import datetime
from collections import defaultdict
#from scipy.sparse import *
from scipy.sparse.linalg import norm

narticles = 1000
max_wordid = 0 # this is 'k' for any vector computation - jk no, take advantage of sparsity

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

    #with open("p2_data/data50.csv") as f:
    #    data = [list(map(int, d.strip().split(","))) for d in f]

    #article_ids = (np.array(article_ids) - 1).astype(np.int)
    #word_ids = (np.array(word_ids) - 1).astype(np.int)
    #sparse_counts = coo_matrix((counts, (article_ids, word_ids)))


    print("max_wordid =", max_wordid)
    return data

#def expand_vector(V):
#    V_expanded = np.zeros((max_wordid, 1))
#    for word_pair in V:
#        V_expanded[word_pair[0]] = word_pair[1]
#    return V_expanded

def flatten_vector(V): # get the frequencies only from the word pairs
    V_flattened = np.zeros((len(V), 1))
    for i in range(len(V)):
        V_flattened[i] = V[i][1]
    return V_flattened

def hyperplane_hash(d, V):
    # Populate (d x k) matrix from Gaussian distribution
    mu, sigma = 0, 1
    Mi = np.random.normal(mu, sigma, (d, len(V)))

    # Get (d x 1) vector MiV
    MiV = np.dot(Mi, flatten_vector(V))

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

# Runtime Optimizations Options:
# - ***for each such computation we can (similar to previous parts) exploit the sparsity of the vectors and avoid going through all the 60,000+ dimensions.
#       (https://piazza.com/class/k8gkb66j2i53qe?cid=185)
# - save the hash values for each article into a dictionary of lists.
#       (https://piazza.com/class/k8gkb66j2i53qe?cid=188)
# - More efficiently convert bin to dec
#       (https://piazza.com/class/k8gkb66j2i53qe?cid=190)

def get_best_candidate(candidates, Q, dataset):
    best_score = (0,0)
    first = True
    for c_idx in candidates:
        C = dataset[int(c_idx)]



        # Get sparse cosine calculation between Q and C
        print("max =", max_wordid)
        word_ids, cnts = list(zip(*Q))
        coo1 = coo_matrix((cnts, ([0]*len(word_ids), word_ids)), shape=(1, max_wordid+1))
        word_ids, cnts = list(zip(*C))
        coo2 = coo_matrix((cnts, (word_ids, [0]*len(word_ids))), shape=(max_wordid+1, 1))
        print(coo1.shape)
        print(coo2.shape)
        cosine_sim = coo1.dot(coo2)/(linalg.norm(coo1)*linalg.norm(coo2)) # use Cosine similarity
        #cosine_sim = (np.dot(carr, qarr) / (np.linalg.norm(carr) * np.linalg.norm(qarr)))

        # Replace score if better
        score = (c_idx, cosine_sim)
        if first:
            best_score = score
            first = False
        else:
            if (score[1] < best_score[1]):
                best_score = score

    return best_score[0] # return vector index, not actual vector

def query(dataset, buckets_dict, d, l, Q):
    candidates = []

    # Get all candidates
    for i in range(l):
        bucket_idx = hyperplane_hash(d, Q)
        candidates += buckets_dict[bucket_idx]
    candidates = list(set(candidates)) # remove duplicate candidates

    return (len(candidates), get_best_candidate(candidates, Q, dataset)) # return (Sq, most similar candidate's idx)

def query_assess(dataset, buckets_dict, d, l):
    q_idx, err_cnt, tot_cnt, tot_candidates = 0, 0, 0, 0
    for Q in dataset:
        print("querying article", q_idx)
        if (q_idx%25 == 0):
            dateTimeObj = datetime.now()
            print("current time =", dateTimeObj)
            print("err_cnt =", err_cnt, " ; tot_cnt =", tot_cnt)

        (cur_candidates, match_idx) = query(dataset, buckets_dict, d, l, Q)
        if (q_idx != match_idx):
            err_cnt += 1
        q_idx += 1
        tot_cnt += 1
        tot_candidates += cur_candidates

    err_perc = err_cnt/tot_cnt
    avg_candidates = tot_candidates/narticles
    return (err_perc, avg_candidates)

def read_buckets_csv(d):
    buckets_dict = defaultdict(list)
    in_csv = "buckets/buckets_d" + str(d) + ".csv"
    with open(in_csv, 'rb') as csvfile:
        buckets_csv = csv.reader(csvfile, delimiter=',')
        idx = 0
        for row in buckets_csv:
            buckets_dict[idx] = row
            idx += 1
    return buckets_dict

def experiment(d, l, dataset):
    buckets_dict = read_buckets_csv(d)
    return query_assess(dataset, buckets_dict, d, l)

def main():
    l = 128                     # Number of hash tables
    d_lst = list(range(5,21))   # Log2(# buckets)
    dataset = import_data()     # dataset = list of lists

    # Run the experiment for each d
    err_perc_lst = []
    avg_candidates_lst = []
    for cur_d in d_lst:
        dateTimeObj = datetime.now()
        print("current time =", dateTimeObj)
        print("cur_d =", cur_d)
        (err_perc, avg_candidates) = experiment(cur_d, l, dataset)
        err_perc_lst.append(err_perc)
        avg_candidates_lst.append(avg_candidates)
        print("err_perc =", err_perc)
        print("avg_candidates =", avg_candidates)

    # plot err_cnt vs Sq_size
    plt.plot(err_perc_lst, avg_candidates_lst)
    plt.show()


main()
