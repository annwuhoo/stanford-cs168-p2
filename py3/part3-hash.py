# Project: CS 168 SP2020 - Miniproject 2 - Question 3
# Author: Ann Wu
# Date: 04/20/2020

import numpy as np
import matplotlib.pyplot as plt
import sys
import csv
from datetime import datetime

narticles = 1000
max_wordid = 0  # this is 'k' for any vector computation

# Separate the data into each article's bag of words, where each word = (x,y) where x = wordID and y = freq
def import_data():
    global max_wordid

    data = [[] for i in range(narticles)]  # init list of lists for data
    with open("p2_data/data50.csv", "rb") as csvfile:
        data50reader = csv.reader(csvfile, delimiter=",")
        for row in data50reader:
            data[int(row[0]) - 1].append((int(row[1]), int(row[2])))
            if int(row[1]) > max_wordid:
                max_wordid = int(row[1])

    print("max_wordid =", max_wordid)
    return data

def flatten_vector(V):  # get the frequencies only from the word pairs
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
    MiV_sgn[MiV_sgn == -1] = 0  # Replace all elements with value -1 with 0
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
        print("mapping article", vector_idx)
        if vector_idx % 25 == 0:
            dateTimeObj = datetime.now()
            print("current time =", dateTimeObj)
            for bucket in buckets:
                print(bucket)

        # for each iter, hash the element and insert the current vector_idx into the hash bucket
        for i in range(l):
            bucket_idx = hyperplane_hash(d, V)
            buckets[bucket_idx].append(vector_idx)
        vector_idx += 1

        # Remove duplicates in each bucket
        for i in range(2 ** d):
            tmp = list(set(buckets[i]))
            buckets[i] = tmp

    return


def write_buckets_csv(buckets, d):
    out_csv = "buckets/buckets_d" + str(d) + ".csv"
    with open(out_csv, "wb") as csvfile:
        buckets_csv = csv.writer(csvfile, delimiter=",")
        for bucket in buckets:
            buckets_csv.writerow(bucket)

def experiment(d, l, dataset):
    buckets = [[] for i in range(2 ** d)]  # init list of lists for Classification
    populate_buckets(dataset, buckets, d, l)
    write_buckets_csv(buckets, d)
    # return query_assess(dataset, buckets, d, l)


def main():
    l = 128  # Number of hash tables
    d_lst = list(range(5, 21))  # Log2(# buckets)
    dataset = import_data()  # dataset = list of lists

    # Run the experiment for each d
    # err_perc_lst = []
    # num_candidates_lst = []
    for cur_d in d_lst:
        dateTimeObj = datetime.now()
        print("current time =", dateTimeObj)
        print("cur_d =", cur_d)
        experiment(cur_d, l, dataset)
        # (err_perc, num_candidates) = experiment(cur_d, l, dataset)
        # err_perc_lst.append(err_perc)
        # num_candidates_lst.append(num_candidates)

    # plot err_cnt vs Sq_size
    # plt.plot(err_perc_lst, num_candidates_lst)
    # plt.show()


main()
