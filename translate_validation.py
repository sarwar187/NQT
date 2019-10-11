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
from Batch import nopeak_mask
import math
import numpy as np
from pathlib import Path
import subprocess
from nltk.translate.bleu_score import sentence_bleu, corpus_bleu
from Models import Transformer
from translation.translator import Translator
from translation.query import *

def tokenizer(text):  # create a tokenizer function
    return text.split()

def create_fields(opt):
    #print("loading tokenizers...")
    SRC = pickle.load(open(f'{opt.load_vocab}/SRC.pkl', 'rb'))
    TRG = pickle.load(open(f'{opt.load_vocab}/TRG.pkl', 'rb'))
    return (SRC, TRG)

def evaluate():
    # Check current working directory.
    # retval = os.getcwd()
    os.chdir("CLEF-ENG-ML-NEW/")
    p = subprocess.Popen(
        '/cm/shared/apps/java/jdk1.8.0_191/bin/java -cp target/lib/*:target/CLEF-ENG-ML-1.0-SNAPSHOT.jar rabflair.flair.myBatchSearch',
        shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    line = p.stdout.readlines()[0]
    line_splitted = line.decode().split()
    mAP = float(line_splitted[3])
    # retval = p.wait()
    os.chdir("../")
    # print("percentage of uncovered terms " + str(sum_percentage_uncovered / length))
    # print("corpus bleu score is " + str(corpus_bleu(list_of_references, hypotheses)) + " and average MAP is " + str(mAP))
    print("MAP:\t" + str(mAP))
    # print("BLEU:\t" + str(corpus_bleu(list_of_references, hypotheses)))

parser = argparse.ArgumentParser()
#If we are working on small dataset
small = 0
#If we want to activate relevance based training
relevance_training = 1

if small==1:
    parser.add_argument('-src_data', type=str, default='data/italian_small.txt')
    parser.add_argument('-trg_data', type=str, default='data/english_small.txt')
    parser.add_argument('-trg_data_retrieval', type=str, default='data/english_retrieval.txt')

else:
    parser.add_argument('-src_data', type=str, default='data/italian.txt')
    parser.add_argument('-trg_data', type=str, default='data/english.txt')
    parser.add_argument('-trg_data_retrieval', type=str, default='data/LATIMESTEXT2.txt')

parser.add_argument('-src_lang', type=str, default='it')
parser.add_argument('-trg_lang', type=str, default='en')
parser.add_argument('-no_cuda', action='store_true')
parser.add_argument('-SGDR', action='store_true')
parser.add_argument('-epochs', type=int, default=2)
parser.add_argument('-d_model', type=int, default=200)
parser.add_argument('-n_layers', type=int, default=6)
parser.add_argument('-heads', type=int, default=8)
parser.add_argument('-dropout', type=int, default=0.1)
parser.add_argument('-batchsize', type=int, default=1500)
parser.add_argument('-printevery', type=int, default=100)
parser.add_argument('-lr', type=int, default=0.0001)
parser.add_argument('-load_weights', type=str, default='weights')
parser.add_argument('-load_vocab', type=str, default='clir_it_en')
#parser.add_argument('-load_weights', type=str, default='tiny_train')
parser.add_argument('-create_valset', action='store_true')
parser.add_argument('-max_strlen', type=int, default=80)
parser.add_argument('-floyd', action='store_true')
parser.add_argument('-checkpoint', type=int, default=0)

os.chdir("/mnt/nfs/work1/allan/smsarwar/material/pytorch_transformer/")
opt = parser.parse_args(args=[])
opt.device = 0 if opt.no_cuda is False else -1
opt.max_len = 20
opt.k=1

#print(opt)
SRC, TRG = create_fields(opt)
# opt.train = create_dataset(opt, SRC, TRG)
translator = Translator(SRC, TRG)

model = Transformer(len(SRC.vocab), len(TRG.vocab), opt.d_model, opt.n_layers, opt.heads, opt.dropout, 2)
model = model.cuda()
model.load_state_dict(torch.load(f'{opt.load_weights}/model_weights'))

#stemmer = SnowballStemmer("english")

# Query dict created
# Order in which queries are put: title, title_description, description
# load italian queries
eng_query_dict = translator.load_query_file("CLEF-ENG-ML/index/english/all", "All-eng-tit-final.tsv",                               "All-eng-tit-des-final-clef.tsv")
it_query_dict = translator.load_query_file("CLEF-ENG-ML/index/italian", "All-Top-ita-tit-final.txt",
                                "All-Top-ita-tit-desc-final.txt")
it_query_tt_dict = translator.load_query_tt("CLEF-ENG-ML/index/italian", "italian_tt.txt")
translation_file = open(os.path.join("CLEF-ENG-ML-NEW/index/english/all", "All-eng-tit-des-final.tsv"), "w")

sum_percentage_uncovered = 0.0
list_of_references = []
hypotheses = []
length = 0

for key in sorted(it_query_dict)[0:50]:
    length += 1
    opt.text = it_query_dict[key][1]  ###
    tt_query = it_query_tt_dict[opt.text]
    candidate, queries, query_tokens = translator.translate(opt, model, SRC, TRG)
    candidate = candidate.lower()  ###
    reference = eng_query_dict[key][1]  ###
    list_of_references.append(reference)
    hypotheses.append(candidate)
    reference_splitted = set(reference.split())
    candidate_splitted = set(candidate.split())
    uncovered = reference_splitted.difference(candidate_splitted)
    percentage_uncovered = len(uncovered) / (len(reference_splitted) * 1.0)
    sum_percentage_uncovered += percentage_uncovered
    covered = reference_splitted.difference(uncovered)
    str_uncovered = ' '.join(uncovered)
    str_covered = ' '.join(covered)
    translation_file.write(key + "\t" + candidate + " \n")

translation_file.close()
evaluate()
