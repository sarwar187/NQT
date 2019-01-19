#file:client.py
from jsonsocket import Client
import time
import socket
import json
import math
import numpy as np
import multiprocessing as mp

class SuperClient():
    def __init__(self):
        self.server_names = []
        for line in open("server_names.txt"):
            self.server_names.append(line)
        self.hosts = []
        self.ports = []
        for server_name in self.server_names:
            input_splitted = server_name.split(":")
            self.hosts.append(input_splitted[0])
            self.ports.append(int(input_splitted[1]))

    def list_to_dictionary(self, list):
        dict = {}
        for index, item in enumerate(list):
            dict[index] = item
        return dict

    def dictionary_to_list(self, dictionary):
        values = []
        keylist = sorted(dictionary)
        for key in keylist:
            values.append(dictionary[key])
        return values

    def execute_on_server(self, query_list_subset, host, port):
        client = Client()
        query_dict = self.list_to_dictionary(query_list_subset)
        client.connect(host, port).send(query_dict)
        response = client.recv()
        self.output.put(response)

    def load_servers(self):
        server_names = []
        for line in open("server_names.txt"):
            server_names.append(line)
        hosts = []
        ports = []
        for server_name in server_names:
            input_splitted = server_name.split(":")
            hosts.append(input_splitted[0])
            ports.append(int(input_splitted[1]))
        return server_names, hosts, ports

    def query_expansion_distributed(self, query_list):
        number_of_servers = len(self.hosts)
        #print(number_of_servers)
        length = math.floor(len(query_list) / number_of_servers)
        remainder = len(query_list) % number_of_servers
        size_array = [length for i in np.arange(number_of_servers)]

        for i in np.arange(remainder):
            size_array[i]+=1
        #print(size_array)
        results = []
        processes = []
        #print(len(size_array))

        self.output = mp.Queue()
        for i in np.arange(len(size_array)):
            query_list_subset = query_list[i * length: i * length + size_array[i]]
            processes.append(mp.Process(target=self.execute_on_server, args=(query_list_subset, self.hosts[i], self.ports[i])))
            # Run processes
            processes[i].start()
            # Exit the completed processes
        for p in processes:
            p.join()
            # Get process results from the output queue
        while not self.output.empty():
            results.append(self.output.get())  # as docs say: Remove and return an item from the queue.

        final_result = []
        for host in self.hosts:
            for result in results:
                if host == list(result.keys())[0]:
                    returned_dictionary = result[host]
                    final_result.extend(self.dictionary_to_list(returned_dictionary))

        return final_result


def main():
    #load query file
    #batch_size = sy

    query_list = ["european crime records", "crime records", "european crime", "european records", "crime crimes violent sheriff enforcement re criminals stresak bill strikes", "LA times corpus", "crime records"]
    #query_list = ["european crime records" for i in range(500)]

    super_client = SuperClient()
    print(super_client.hosts)
    final_result = super_client.query_expansion_distributed(query_list)
    print(final_result)

if __name__ == '__main__':
    main()
