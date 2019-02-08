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

def get_synonym(word, SRC):
    syns = wordnet.synsets(word)
    for s in syns:
        for l in s.lemmas():
            if SRC.vocab.stoi[l.name()] != 0:
                return SRC.vocab.stoi[l.name()]

    return 0


def multiple_replace(dict, text):
    # Create a regular expression  from the dictionary keys
    regex = re.compile("(%s)" % "|".join(map(re.escape, dict.keys())))

    # For each match, look-up corresponding value in dictionary
    return regex.sub(lambda mo: dict[mo.string[mo.start():mo.end()]], text)


def translate_sentence(sentence, model, opt, SRC, TRG):
    model.eval()
    indexed = []
    sentence = SRC.preprocess(sentence)
    for tok in sentence:
        if SRC.vocab.stoi[tok] != 0 or opt.floyd == True:
            indexed.append(SRC.vocab.stoi[tok])
        else:
            indexed.append(get_synonym(tok, SRC))
    sentence = Variable(torch.LongTensor([indexed]))
    if opt.device == 0:
        sentence = sentence.cuda()

    sentence = beam_search(sentence, model, SRC, TRG, opt)

    return multiple_replace({' ?': '?', ' !': '!', ' .': '.', '\' ': '\'', ' ,': ','}, sentence)


def translate(opt, model, SRC, TRG):
    sentences = opt.text.lower().split('.')
    translated = []

    for sentence in sentences:
        translated.append(translate_sentence(sentence + '.', model, opt, SRC, TRG).capitalize())

    return (' '.join(translated))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-load_weights', required=True)
    parser.add_argument('-k', type=int, default=3)
    parser.add_argument('-max_len', type=int, default=80)
    parser.add_argument('-d_model', type=int, default=512)
    parser.add_argument('-n_layers', type=int, default=6)
    parser.add_argument('-src_lang', required=True)
    parser.add_argument('-trg_lang', required=True)
    parser.add_argument('-heads', type=int, default=8)
    parser.add_argument('-dropout', type=int, default=0.1)
    parser.add_argument('-no_cuda', action='store_true')
    parser.add_argument('-floyd', action='store_true')
    parser.add_argument('-qdir', required=True)
    parser.add_argument('-qtitle', required=True)
    parser.add_argument('-qdesc', required=True)
    parser.add_argument('-approach', type=str, default="baseline")

    opt = parser.parse_args()

    opt.device = 0 if opt.no_cuda is False else -1

    assert opt.k > 0
    assert opt.max_len > 10

    #Creating query_dict where title and query against a query id would be available
    query_file_title = open(os.path.join(opt.qdir, opt.qtitle))
    query_dict = {}
    query_title_len_dict = {}
    for line in query_file_title:
        line_splitted = line.split("\t")
        query_id = line_splitted[0].strip()
        query_title = line_splitted[1].strip()
        query_dict.setdefault(query_id, [])
        query_dict[query_id].append(query_title)
        query_title_len_dict[query_id] = len(query_title.split())
        #print(line)

    query_file_desc = open(os.path.join(opt.qdir, opt.qdesc))
    for line in query_file_desc:
        line_splitted = line.split("\t")
        query_id = line_splitted[0].strip()
        query_title_desc = line_splitted[1].strip()
        query_dict[query_id].append(query_title_desc)
        title_len = query_title_len_dict[query_id]
        query_desc = ' '.join(query_title_desc.split()[title_len: ])
        query_dict[query_id].append(query_desc)

    #Query dict created
    #Order in which queries are put: title, title_description, description
    for line in query_file_desc:
        print(line)
    print(query_file_desc)

    translation_file = open(os.path.join(opt.qdir, "nqt", "translation_baseline_epoch5.txt"), "w")

    SRC, TRG = create_fields(opt)
    model = get_model(opt, len(SRC.vocab), len(TRG.vocab))
    phrase = "query"
    for query_id in query_dict.keys():
        #translate the title first
        opt.text = query_dict[query_id][0]
        print("1---------------------------------------------------")
        print(opt.text)
        phrase = translate(opt, model, SRC, TRG)
        query_dict[query_id].append(phrase)
        print(phrase)
        print("***************************************************")
        # now translate the title_description

        opt.text = query_dict[query_id][1]
        print("2---------------------------------------------------")
        print(opt.text)
        phrase = translate(opt, model, SRC, TRG)
        query_dict[query_id].append(phrase)
        # now translate the description only
        print(phrase)
        print("***************************************************")

        opt.text = query_dict[query_id][2]
        print("3---------------------------------------------------")
        print(opt.text)
        phrase = translate(opt, model, SRC, TRG)
        query_dict[query_id].append(phrase)
        print(phrase)
        print("***************************************************")

    json.dump(query_dict, translation_file)


if __name__ == '__main__':
    main()
