import argparse
import time
import torch
from Models import get_model
from Process import *
import torch.nn.functional as F
from Optim import CosineWithRestarts
from Batch import create_masks
import pdb
import dill as pickle
import argparse
from Models import get_model
from Beam import beam_search
from nltk.corpus import wordnet
from torch.autograd import Variable
import re
import json
import statistics
from nltk.stem.snowball import SnowballStemmer


class Translator():

    def __init__(self, SRC, TRG):
        self.SRC = SRC
        self.TRG = TRG

    def get_synonym(self, word, SRC):
        syns = wordnet.synsets(word)
        for s in syns:
            for l in s.lemmas():
                if SRC.vocab.stoi[l.name()] != 0:
                    return SRC.vocab.stoi[l.name()]

        return 0

    def multiple_replace(self, dict, text):
        # Create a regular expression  from the dictionary keys
        regex = re.compile("(%s)" % "|".join(map(re.escape, dict.keys())))

        # For each match, look-up corresponding value in dictionary
        return regex.sub(lambda mo: dict[mo.string[mo.start():mo.end()]], text)

    def preprocess_sentence(self, sentence, model, opt, SRC, TRG):
        indexed = []
        sentence = SRC.preprocess(sentence)
        for tok in sentence:
            if SRC.vocab.stoi[tok] != 0 or opt.floyd == True:
                indexed.append(SRC.vocab.stoi[tok])
            else:
                indexed.append(self.get_synonym(tok, SRC))
        sentence = Variable(torch.LongTensor([indexed]))

    def translate_sentence(self, sentence, model, opt, SRC, TRG):
        model.eval()
        indexed = []
        sentence = SRC.preprocess(sentence)
        for tok in sentence:
            if SRC.vocab.stoi[tok] != 0 or opt.floyd == True:
                indexed.append(SRC.vocab.stoi[tok])
            else:
                indexed.append(self.get_synonym(tok, SRC))
        sentence = Variable(torch.LongTensor([indexed]))
        if opt.device == 0:
            sentence = sentence.cuda()

        sentences, query, string_query = beam_search(sentence, model, SRC, TRG, opt)
        # print(sentences)
        # print(query)

        for sentence in sentences:
            self.multiple_replace({' ?': '?', ' !': '!', ' .': '.', '\' ': '\'', ' ,': ','}, sentence)
        return sentences, query, string_query

    def translate(self, opt, model, SRC, TRG):
        sentences = opt.text.lower().split('.')
        translated = []
        queries = []
        # print(translate_sentence(sentence + '.', model, opt, SRC, TRG))
        for sentence in sentences:
            translated_sentences, query, string_query = self.translate_sentence(sentence + '.', model, opt, SRC, TRG)
            for translated_sentence in translated_sentences:
                translated.append(translated_sentence.capitalize())
            queries.append(query)

        return (' '.join(translated)), queries, string_query


    def load_query_tt(self, qdir, tt_file):
        #passing query directory, query title and query title description
        #Creating query_dict where title and query against a query id would be available
        query_tt_dict = {}
        query_file_tt = open(os.path.join(qdir, tt_file))
        for line in query_file_tt:
            if len(line) > 1:
            #print(line)
                line_splitted = line.split("\t")
                query_title = line_splitted[1].strip()
                query_translation_tt = line_splitted[2].strip()
                query_tt_dict.setdefault(query_title, query_translation_tt)
        return query_tt_dict

    def load_query_file(self, qdir, qtitle, qdesc):
        # Creating query_dict where title and query against a query id would be available
        query_file_title = open(os.path.join(qdir, qtitle))
        query_dict = {}
        query_title_len_dict = {}
        for line in query_file_title:
            line_splitted = line.split("\t")
            query_id = line_splitted[0].strip()
            query_title = line_splitted[1].strip()
            query_dict.setdefault(query_id, [])
            query_dict[query_id].append(query_title)
            query_title_len_dict[query_id] = len(query_title.split())
            # print(line)
        query_file_desc = open(os.path.join(qdir, qdesc))
        for line in query_file_desc:
            line_splitted = line.split("\t")
            query_id = line_splitted[0].strip()
            query_title_desc = line_splitted[1].strip()
            query_dict[query_id].append(query_title_desc)
            title_len = query_title_len_dict[query_id]
            query_desc = ' '.join(query_title_desc.split()[title_len:])
            query_dict[query_id].append(query_desc)
        return query_dict
    
