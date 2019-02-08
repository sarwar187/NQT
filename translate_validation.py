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
os.chdir("/mnt/nfs/work1/allan/smsarwar/material/pytorch_transformer/")

parser = argparse.ArgumentParser()
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

#-load_weights clir_it_en -src_lang it -trg_lang en -k 2
#opt = parser.parse_args()
opt = parser.parse_args(args=[])
opt.device = 0 if opt.no_cuda is False else -1
#print(opt)
#assert opt.k > 0
#assert opt.max_len > 10

def tokenizer(text):  # create a tokenizer function
    return text.split()


def create_fields(opt):
    #print("loading tokenizers...")
    TRG = data.Field(lower=True, tokenize=tokenizer, init_token='<sos>', eos_token='<eos>')
    SRC = data.Field(lower=True, tokenize=tokenizer)
    SRC = pickle.load(open(f'{opt.load_vocab}/SRC.pkl', 'rb'))
    TRG = pickle.load(open(f'{opt.load_vocab}/TRG.pkl', 'rb'))
    return (SRC, TRG)


def create_dataset(opt, SRC, TRG):
    #print("creating dataset and iterator... ")

    raw_data = {'src': [line for line in open(opt.src_data)], 'trg': [line for line in open(opt.trg_data)]}
    df = pd.DataFrame(raw_data, columns=["src", "trg"])

    mask = (df['src'].str.count(' ') < opt.max_strlen) & (df['trg'].str.count(' ') < opt.max_strlen)
    df = df.loc[mask]

    df.to_csv("translate_transformer_temp.csv", index=False)

    data_fields = [('src', SRC), ('trg', TRG)]
    train = data.TabularDataset('./translate_transformer_temp.csv', format='csv', fields=data_fields)

    train_iter = MyIterator(train, batch_size=opt.batchsize, device=opt.device,
                            repeat=False, sort_key=lambda x: (len(x.src), len(x.trg)),
                            batch_size_fn=batch_size_fn, train=True, shuffle=True)

    os.remove('translate_transformer_temp.csv')

    #print("creating target vocabulary ... ")
    raw_data = {'trg': [line for line in open(opt.trg_data_retrieval)]}
    df = pd.DataFrame(raw_data, columns=["trg"])
    mask = (df['trg'].str.count(' ') > 1)
    df = df.loc[mask]
    df.to_csv("translate_transformer_retrieval_temp.csv", index=False)
    data_fields = [('trg', TRG)]
    train_retrieval = data.TabularDataset('./translate_transformer_retrieval_temp.csv', format='csv',
                                          fields=data_fields)
    os.remove('translate_transformer_retrieval_temp.csv')
    SRC.build_vocab(train)
    TRG.build_vocab(train, train_retrieval)
    # TRG.build_vocab(train_temp)
    if opt.checkpoint > 0:
        try:
            os.mkdir("weights")
        except:
            print("weights folder already exists, run program with -load_weights weights to load them")
            quit()
        pickle.dump(SRC, open('weights/SRC.pkl', 'wb'))
        pickle.dump(TRG, open('weights/TRG.pkl', 'wb'))
    opt.src_pad = SRC.vocab.stoi['<pad>']
    opt.trg_pad = TRG.vocab.stoi['<pad>']
    opt.train_len = get_len(train_iter)
    return train_iter


#print(opt)
SRC, TRG = create_fields(opt)
# opt.train = create_dataset(opt, SRC, TRG)

from Models import Transformer

model = Transformer(len(SRC.vocab), len(TRG.vocab), opt.d_model, opt.n_layers, opt.heads, opt.dropout)
model = model.cuda()
model.load_state_dict(torch.load(f'{opt.load_weights}/model_weights'))
#model.load_state_dict(torch.load(f'{opt.load_weights}/model_weights_best_validation'))
# model.load_state_dict(torch.load(f'{opt.load_weights}/model_weights_best'))
# model.load_state_dict(torch.load(f'{opt.load_weights}/model_weights_2_epoch_large_full'))
# model = get_model(opt, len(SRC.vocab), len(TRG.vocab))
opt.k = 3
opt.max_len = 20


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


def preprocess_sentence(sentence, model, opt, SRC, TRG):
    indexed = []
    sentence = SRC.preprocess(sentence)
    for tok in sentence:
        if SRC.vocab.stoi[tok] != 0 or opt.floyd == True:
            indexed.append(SRC.vocab.stoi[tok])
        else:
            indexed.append(get_synonym(tok, SRC))
    sentence = Variable(torch.LongTensor([indexed]))


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

    sentences, query = beam_search(sentence, model, SRC, TRG, opt)
    # print(sentences)
    # print(query)

    for sentence in sentences:
        multiple_replace({' ?': '?', ' !': '!', ' .': '.', '\' ': '\'', ' ,': ','}, sentence)
    return sentences, query


def translate(opt, model, SRC, TRG):
    sentences = opt.text.lower().split('.')
    translated = []
    queries = []
    # print(translate_sentence(sentence + '.', model, opt, SRC, TRG))
    for sentence in sentences:
        translated_sentences, query = translate_sentence(sentence + '.', model, opt, SRC, TRG)
        for translated_sentence in translated_sentences:
            translated.append(translated_sentence.capitalize())
        queries.append(query)

    return (' '.join(translated)), queries


def init_vars(src, model, SRC, TRG, opt):
    init_tok = TRG.vocab.stoi['<sos>']
    src_mask = (src != SRC.vocab.stoi['<pad>']).unsqueeze(-2)
    # this is the output from the encoder
    e_output = model.encoder(src, src_mask)
    # this is initializing the outputs
    outputs = torch.LongTensor([[init_tok]])
    if opt.device == 0:
        outputs = outputs.cuda()

    trg_mask = nopeak_mask(1, opt)
    src_mask = src_mask.cuda()
    trg_mask = trg_mask.cuda()
    outputs = outputs.cuda()
    e_output = e_output.cuda()

    out = model.out(model.decoder(outputs,
                                  e_output, src_mask, trg_mask))
    out = F.softmax(out, dim=-1)

    probs, ix = out[:, -1].data.topk(opt.k)
    preds_token_ids = ix.view(ix.size(0), -1)
    pred_strings = [' '.join([TRG.vocab.itos[ind] for ind in ex]) for ex in preds_token_ids]

    # print (pred_strings)

    log_scores = torch.Tensor([math.log(prob) for prob in probs.data[0]]).unsqueeze(0)

    outputs = torch.zeros(opt.k, opt.max_len).long()
    if opt.device == 0:
        outputs = outputs.cuda()
    outputs[:, 0] = init_tok
    outputs[:, 1] = ix[0]

    e_outputs = torch.zeros(opt.k, e_output.size(-2), e_output.size(-1))
    if opt.device == 0:
        e_outputs = e_outputs.cuda()
    e_outputs[:, :] = e_output[0]

    return outputs, e_outputs, log_scores


def k_best_outputs(outputs, out, log_scores, i, k):
    probs, ix = out[:, -1].data.topk(k)

    preds_token_ids = ix.view(ix.size(0), -1)
    preds_probs = probs.view(probs.size(0), -1)

    pred_strings = [' '.join([TRG.vocab.itos[ind] for ind in ex]) for ex in preds_token_ids]
    # print (pred_strings)
    # pred_probs_string = [' '.join([str(ex[ind]) for ind in ex]) for ex in preds_probs]
    pred_strings = []
    pred_strings_dict = {}
    for pred_token_id, prob in zip(preds_token_ids, preds_probs):
        pred_strings_temp = ''
        for iid, prob in zip(pred_token_id, prob):
            prob = prob.item()
            if prob > 0.01:
                pred_strings_temp += str(TRG.vocab.itos[iid]) + ' '
            if str(TRG.vocab.itos[iid]) in pred_strings_dict:
                if prob > pred_strings_dict[str(TRG.vocab.itos[iid])]:
                    pred_strings_dict[str(TRG.vocab.itos[iid])] = prob
            else:
                pred_strings_dict[str(TRG.vocab.itos[iid])] = prob

        pred_strings.append(pred_strings_temp)
    # print (preds_probs)
    # print (pred_probs_strings)

    # print("indices of top k")
    # print(ix)
    log_probs = torch.Tensor([math.log(p) for p in probs.data.view(-1)]).view(k, -1) + log_scores.transpose(0, 1)
    k_probs, k_ix = log_probs.view(-1).topk(k)

    row = k_ix // k
    col = k_ix % k

    outputs[:, :i] = outputs[row, :i]
    outputs[:, i] = ix[row, col]

    log_scores = k_probs.unsqueeze(0)

    return outputs, log_scores, pred_strings, pred_strings_dict


def beam_search(src, model, SRC, TRG, opt):
    outputs, e_outputs, log_scores = init_vars(src, model, SRC, TRG, opt)
    eos_tok = TRG.vocab.stoi['<eos>']
    src_mask = (src != SRC.vocab.stoi['<pad>']).unsqueeze(-2)
    ind = None
    query = {}
    for i in range(2, opt.max_len):

        trg_mask = nopeak_mask(i, opt)
        src_mask = src_mask.cuda()
        trg_mask = trg_mask.cuda()

        out = model.out(model.decoder(outputs[:, :i], e_outputs, src_mask, trg_mask))

        out = F.softmax(out, dim=-1)

        # print("output data shape")
        # print(out.data.shape)

        outputs, log_scores, pred_strings, pred_strings_dict = k_best_outputs(outputs, out, log_scores, i, opt.k)

        #         This part is another way of forming the query dictionary
        for pred_string in pred_strings:
            pred_string_splitted = pred_string.split()
            for st in pred_string_splitted:
                query.setdefault(st, 1.0)
                query[st] = query[st] + 1

        #         for term in pred_strings_dict:
        #             if term in query:
        #                 if pred_strings_dict[term] > query [term]:
        #                     query[term] = pred_strings_dict[term]
        #             else:
        #                 query[term] = pred_strings_dict[term]

        if (outputs == eos_tok).nonzero().size(0) == opt.k:
            alpha = 0.7
            div = 1 / ((outputs == eos_tok).nonzero()[:, 1].type_as(log_scores) ** alpha)
            _, ind = torch.max(log_scores * div, 1)
            ind = ind.data[0]
            break
    # print("query")
    # print(query)
    # if ind is None:
    #     length = (outputs[0] == eos_tok).nonzero()[0]
    #     return ' '.join([TRG.vocab.itos[tok] for tok in outputs[0][1:length]])
    #
    # else:
    #     length = (outputs[ind] == eos_tok).nonzero()[0]
    # return ' '.join([TRG.vocab.itos[tok] for tok in outputs[ind][1:length]])

    if ind is None:
        query_list = []
        # print("value of k is " + str(opt.k))
        for i in np.arange(opt.k):
            if eos_tok in outputs[i]:
                length = (outputs[i] == eos_tok).nonzero()[0]
            else:
                length = opt.max_len
            query_list.append(' '.join([TRG.vocab.itos[tok] for tok in outputs[i][1:length]]))
        return query_list, query

        # if (outputs[0]==eos_tok).nonzero().size(0) >= 1:
        #     length = (outputs[0]==eos_tok).nonzero()[0]
        #     return ' '.join([TRG.vocab.itos[tok] for tok in outputs[0][1:length]])
        # else:
        #     return ' '


    else:
        # if (outputs[ind] == eos_tok).nonzero().size(0) >= 1:
        #     length = (outputs[ind]==eos_tok).nonzero()[0]
        #     return ' '.join([TRG.vocab.itos[tok] for tok in outputs[ind][1:length]])
        # else:
        #     return ' '
        query_list = []
        # print("value of k is " + str(opt.k))
        for i in np.arange(opt.k):
            if eos_tok in outputs[i]:
                length = (outputs[i] == eos_tok).nonzero()[0]
            else:
                length = opt.max_len
            query_list.append(' '.join([TRG.vocab.itos[tok] for tok in outputs[i][1:length]]))
        return query_list, query


def create_galago_query(query_dict, text):
    # print("printing query dict")
    # print(query_dict)
    probs = [query_dict[key] for key in query_dict]
    mean = statistics.mean(probs)

    text_splitted = text.split()
    for token in text_splitted:
        if token not in query_dict:
            query_dict[token] = mean
    probs = [query_dict[key] for key in query_dict]
    sm = sum(probs)
    new_query_dict = {}
    for key in query_dict:
        query_dict[key] /= sm
        if (query_dict[key] > 0.01):
            new_query_dict[key] = query_dict[key]
    query_dict = new_query_dict
    st = "#combine:"
    query_keys = query_dict.keys()
    for index, val in enumerate(query_keys):
        st += str(index) + "=" + str(query_dict[val])
        st += ":"
    st = st.rstrip(":")
    st += "("
    for index, val in enumerate(query_keys):
        st += str(val + " ")
    st += ")"
    return st

import statistics
from nltk.stem.snowball import SnowballStemmer
stemmer = SnowballStemmer("english")
opt.k=1
opt.text = "epidemia ebola zaire recupera documenti parlano misure preventive prese dopo scoppio epidemia ebola zaire"

def load_query_tt(qdir, tt_file):
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

def load_query_file(qdir, qtitle, qdesc):
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


# Query dict created
# Order in which queries are put: title, title_description, description
# load italian queries
eng_query_dict = load_query_file("CLEF-ENG-ML/index/english/all", "All-eng-tit-final.tsv",
                                 "All-eng-tit-des-final-clef.tsv")
it_query_dict = load_query_file("CLEF-ENG-ML/index/italian", "All-Top-ita-tit-final.txt",
                                "All-Top-ita-tit-desc-final.txt")
it_query_tt_dict = load_query_tt("CLEF-ENG-ML/index/italian", "italian_tt.txt")
translation_file = open(os.path.join("CLEF-ENG-ML-NEW/index/english/all", "All-eng-tit-des-final.tsv"), "w")
sum_percentage_uncovered = 0.0
list_of_references = []
hypotheses = []
length = 0
for key in sorted(it_query_dict)[0:50]:
    length += 1
    # print(key)
    opt.text = it_query_dict[key][1]  ###
    tt_query = it_query_tt_dict[opt.text]

    #print(opt.text)
    candidate, queries = translate(opt, model, SRC, TRG)
    #candidate = ' '.join(queries[0].keys())  ###
    #candidate+= ' '.join(queries[1].keys())
    #candidate = candidate + " " + opt.text + " " + tt_query  ### it_query_dict[key][0] #get_entity_strings(opt.text, "it") ###
    candidate = candidate.lower()  ###
    #candidate = create_galago_query(queries[0],
    #                                opt.text)  ### weights estimated by neural approach. The code is put below
    reference = eng_query_dict[key][1]  ###
    list_of_references.append(reference)
    hypotheses.append(candidate)
    reference_splitted = set(reference.split())
    candidate_splitted = set(candidate.split())
    uncovered = reference_splitted.difference(candidate_splitted)
    #print("reference-------------------------------")
    #print(reference)
    #print("candidate-------------------------------")
    #print(candidate)
    #print("uncovered-------------------------------")
    #print(uncovered)
    #print("uncovered%-------------------------------")
    percentage_uncovered = len(uncovered) / (len(reference_splitted) * 1.0)
    sum_percentage_uncovered += percentage_uncovered
    #print(str(percentage_uncovered))
    #print("******************************************")
    covered = reference_splitted.difference(uncovered)
    str_uncovered = ' '.join(uncovered)
    str_covered = ' '.join(covered)

    # translation_file.write(key + "\t" + reference + " \n")
    translation_file.write(key + "\t" + candidate + " \n")
    # translation_file.write(key + "\t" + candidate + " " + reference + " \n")
    # translation_file.write(key + "\t" + " " + str_covered + " \n")
    # translation_file.write(key + "\t" + " " +  + " " + str_uncovered + " \n")
translation_file.close()

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
#print("percentage of uncovered terms " + str(sum_percentage_uncovered / length))
#print("corpus bleu score is " + str(corpus_bleu(list_of_references, hypotheses)) + " and average MAP is " + str(mAP))
print("MAP:\t" + str(mAP))
#print("BLEU:\t" + str(corpus_bleu(list_of_references, hypotheses)))
