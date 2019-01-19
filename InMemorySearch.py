import multiprocessing as mp
import random
import string
import sys
# print("Python version")
# print (sys.version)
# print("Version info.")
# print (sys.version_info)
import os
import whoosh, glob, time, pickle
import whoosh.fields as wf
from whoosh.qparser import QueryParser
from whoosh import index
import threading
from whoosh import filedb
from whoosh.filedb.filestore import FileStorage
import numpy as np
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import multiprocessing as mp
import time


class simple_schema(wf.SchemaClass):
    doc = wf.TEXT(analyzer=whoosh.analysis.StandardAnalyzer(), stored=True)
    filepath = wf.ID(unique=True, stored=True)

class WhooshInMemorySearch():
    def __init__(self):
        #Whoosh index for CLEF
        self.CLEF_index_location = "index/"
        self.CLEF_index_dir = self.CLEF_index_location + "CLEF/"
        # Define an output queue
        self.output = mp.Queue()
        file_storage = FileStorage(self.CLEF_index_dir)
        self.ram_storage = filedb.filestore.copy_to_ram(file_storage)
        self.ix = self.ram_storage.open_index()
        self.reader = self.ix.reader()
        self.results_list = []

    def search(self, query_string_list, field):
        """
        :param query_string_list: list of queries to perform search for with id's
        :param ix: whoosh index for searching
        :param field: the field to search for the keywords
        :return: keywords from a whoosh relevance model
        """
        results_list = []
        for id, query_string in query_string_list:
            #print(query_string)
            query_string_splitted = query_string.split(' ')
            query_string = ' OR '.join(query_string_splitted)
            #print(query_string)
            query = QueryParser(field, self.ix.schema).parse(query_string.encode('latin'))
            with self.ix.searcher() as searcher:
                results = searcher.search(query)
                keywords = [keyword for keyword, score in results.key_terms("doc", docs=10, numterms=10)]
                if len(keywords) < 10:
                    while(len(keywords) != 10):
                        keywords.append("<sos>")
                results_list.append((id, keywords))
                #print(keywords)
        self.output.put(results_list)
        #return results_list

def main():
    #load query file
    #batch_size = sys.argv[1]
    #assigning 100 processes per core
    batch_size = 50
    #we have to divide into sub-batches to send a sub-batch to a core
    small_batch_size = 2
    number_of_batches = int (batch_size/small_batch_size)

    #open the query file
    query_file = open("transQ.tsv")
    query_strings = []
    for line in query_file:
        query_string = line.split("\t")[2]
        query_string_splitted = query_string.split(" ")
        query_string = ' OR '.join(query_string_splitted)
        query_strings.append(query_string)
    #query_strings
    query_strings_tuples = [(i , query_strings[i]) for i in range(batch_size)]

    inmem = WhooshInMemorySearch()
    start_time = time.time()
    # Setup a list of processes that we want to run
    #print ("query string " + query_strings[0:10])
    processes = [mp.Process(target=inmem.search, args=(query_strings_tuples[x * small_batch_size: (x+1) * small_batch_size], "doc")) for x in range(number_of_batches)]
    # Run processes
    for p in processes:
        p.start()
    # Exit the completed processes
    for p in processes:
        p.join()
    # Get process results from the output queue
    results = [inmem.output.get() for p in processes]
    print(results)

    #print(inmem.results_list)
    print("ending multiprocessing")
    print("--- %s seconds ---" % (time.time() - start_time))

if __name__ == '__main__':
    main()
