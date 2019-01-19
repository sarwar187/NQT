from jsonsocket import Server
import time
import socket
import sys
import argparse
import time
from Models import get_model
from Process import *
import torch.nn.functional as F
from Optim import CosineWithRestarts
from Batch import create_masks
import dill as pickle
import time
import math
import numpy as np
import multiprocessing as mp
import random
import string
import sys
import os
import whoosh, glob, time, pickle
import whoosh.fields as wf
from whoosh.qparser import QueryParser
from whoosh import index
import threading
from whoosh import filedb
from whoosh.filedb.filestore import FileStorage
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import time
from InMemorySearch import *
import json

def query_expansion(query_strings_list_all, inmem):
    query_strings_tuples_all = [(i, query_strings_list_all[i][6:-5]) for i in np.arange(len(query_strings_list_all))]

    query_expansion_strings_tuples_all = []
    small_batch_size = 1
    large_batch_size = 25

    length = math.floor(len(query_strings_tuples_all) / large_batch_size)
    remainder = len(query_strings_tuples_all) % large_batch_size
    size_array = [int(large_batch_size) for i in np.arange(length)]
    if remainder!=0:
        size_array.append(int(remainder))

    for i in np.arange(len(size_array)):
        results = []
        query_strings_tuples = query_strings_tuples_all[i * large_batch_size: i * large_batch_size + size_array[i]]
        processes = [mp.Process(target=inmem.search, args=(query_strings_tuples[x * small_batch_size: (x + 1) * small_batch_size], "doc"))
                     for x in np.arange(size_array[i])]

        # Run processes
        for p in processes:
            p.start()
        # Exit the completed processes
        for p in processes:
            p.join()
        # Get process results from the output queue
        while not inmem.output.empty():
            results.append(inmem.output.get())  # as docs say: Remove and return an item from the queue.
        #print("results size " + str(len(results)))
        #results = [inmem.output.get() for p in processes]
        query_expansion_strings_tuples_all.extend(results)
        #print(len(query_expansion_strings_tuples_all))
        #inmem.output.close()
        srt = sorted(query_expansion_strings_tuples_all, key=lambda x: x[0])
    return srt

def main():
    port = int(sys.argv[1])
    host = socket.gethostbyname(socket.gethostname())
    server = Server(host, port)
    print("server started at " + host + ":" + str(port))
    inmem = WhooshInMemorySearch()

    while True:
        server.accept()
        data = server.recv()
        #print(type(data))
        query_dictionary = data
        query_list_tuples = []
        for key in query_dictionary:
            query_list_tuples.append((key, query_dictionary[key]))
        sorted_query_list_tuples = sorted(query_list_tuples, key=lambda x: x[0])
        query_list = [item[1] for item in sorted_query_list_tuples]
        expanded_query_list = query_expansion(query_list, inmem)
        expanded_query_dictionary = {}
        #print(expanded_query_list)
        for item in expanded_query_list:
            #print(item[0][0])
            #print(item[0][1])
            expanded_query_dictionary[str(item[0][0])] = (item[0][1])
        final_result={}
        final_result[host] = expanded_query_dictionary
        server.send(final_result)
        #server.send({"response":data})
    server.close()

main()
