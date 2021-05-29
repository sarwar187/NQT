import argparse
import time
import torch
print(torch.__version__)
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
import numpy as np
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import multiprocessing as mp
import time
# from InMemorySearch import *
from client import *
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from IPython.terminal.debugger import set_trace as keyboard
# from logger import Logger
# logger = Logger('./logs')

parser = argparse.ArgumentParser()
#If we are working on small dataset
small = 1
#If we want to activate relevance based training
relevance_training = 1
dst = 'vocab'
if small==1:
    parser.add_argument('-src_data', type=str, default='../../data/italian_small.txt')
    parser.add_argument('-trg_data', type=str, default='../../data/english_small.txt')
    parser.add_argument('-trg_data_retrieval', type=str, default='../../data/LATIMESTEXT2.txt')
    parser.add_argument('-rm_data', type=str, default='../../data/english_rm_small.txt') ##

else:
    parser.add_argument('-src_data', type=str, default='../../data/italian.txt')
    parser.add_argument('-trg_data', type=str, default='../../data/english.txt')  
    parser.add_argument('-trg_data_retrieval', type=str, default='../../data/LATIMESTEXT2.txt')
    parser.add_argument('-rm_data', type=str, default='../../data/english_rm.txt') ##

parser.add_argument('-src_lang', type=str, default='it')
parser.add_argument('-trg_lang', type=str, default='en')
parser.add_argument('-no_cuda', action='store_true')
parser.add_argument('-SGDR', action='store_true')
parser.add_argument('-epochs', type=int, default=2)
parser.add_argument('-d_model', type=int, default=200)
parser.add_argument('-n_layers', type=int, default=6)
parser.add_argument('-heads', type=int, default=8)
parser.add_argument('-dropout', type=int, default=0.1)
parser.add_argument('-batchsize', type=int, default=1000)
parser.add_argument('-printevery', type=int, default=5)
#if you are running for the first time set load_vocab to None. A vocabulary would be created in the vocab directory. 
parser.add_argument('-load_vocab', type=str, default='vocab')
parser.add_argument('-build_vocab_first_time', default=True)

if relevance_training == 1:
    #if there exists a model_weights file 
    my_file = Path("weights/model_weights")
    if my_file.is_file():
        parser.add_argument('-load_weights', type=str, default='weights')
    else:
        parser.add_argument('-load_weights', type=str, default=None)
    parser.add_argument('-lr', type=int, default=0.01)
else: 
    my_file = Path("weights/model_weights")
    if my_file.is_file():
        parser.add_argument('-load_weights', type=str, default='weights')
    else:
        parser.add_argument('-load_weights', type=str, default=None)         
    parser.add_argument('-lr', type=int, default=0.0001)
    
parser.add_argument('-max_strlen', type=int, default=80)
parser.add_argument('-checkpoint', type=int, default=5)

opt = parser.parse_args(args=[])

def tokenizer(text):  # create a tokenizer function
    return text.split()
    
def create_fields(opt): 
    print("loading tokenizers...") 
    SRC = data.Field(lower=True, tokenize=tokenizer)
    TRG = data.Field(lower=True, tokenize=tokenizer, init_token='<sos>', eos_token='<eos>')
    TRG_REL = data.Field(lower=True, tokenize=tokenizer, init_token='<sos>', eos_token='<eos>')    
    if opt.build_vocab_first_time:
       build_vocab_first_time(opt,SRC, TRG, TRG_REL)
    
    SRC = pickle.load(open(f'{opt.load_vocab}/SRC.pkl', 'rb'))
    TRG = pickle.load(open(f'{opt.load_vocab}/TRG.pkl', 'rb'))
    TRG_REL = pickle.load(open(f'{opt.load_vocab}/TRG.pkl', 'rb'))
    return(SRC, TRG, TRG_REL)

def build_vocab_first_time(opt,SRC, TRG, TRG_REL):
    #if we have not built the vocabulary, we will have to do it here. We are integrating both the retrieval 
    #copus as well as the translation corpus. 
    src_data = [line.strip() for line in opt.src_data]
    trg_data = [line for line in opt.trg_data] 
    raw_data = {'src' : src_data, 'trg': trg_data}
    df_1 = pd.DataFrame(raw_data, columns=["src", "trg"])
    mask = (df_1['src'].str.count(' ') < opt.max_strlen) & (df_1['trg'].str.count(' ') < opt.max_strlen)
    df_1 = df_1.loc[mask]
    df_1.to_csv("vocabulary_1.csv", index=False)
    data_fields = [('src', SRC), ('trg', TRG)]
    train = data.TabularDataset('./vocabulary_1.csv', format='csv', fields=data_fields)
    
    trg_data_retrieval = [line for line in opt.trg_data_retrieval]
    raw_data_retrieval = {'trg': trg_data_retrieval}
    data_fields = [('trg', TRG)]
    df_2 = pd.DataFrame(raw_data, columns=["trg"])
    df_2.to_csv("vocabulary_2.csv", index=False)
    train_retrieval = data.TabularDataset('./vocabulary_2.csv', format='csv', fields=data_fields)
    
    SRC.build_vocab(train)
    TRG.build_vocab(train, train_retrieval)

    # save_vocab(SRC,opt.load_vocab+"SRC.pkl")
    # save_vocab(TRG,opt.load_vocab+"TRG.pkl") 
    # keyboard()()
    pickle.dump(SRC, open(f'{opt.load_vocab}/SRC.pkl', 'wb'))
    pickle.dump(TRG, open(f'{opt.load_vocab}/TRG.pkl', 'wb'))

    os.remove('vocabulary_1.csv')
    os.remove('vocabulary_2.csv')  

def save_vocab(vocab, path):
    import pickle
    output = open(path, 'wb')
    pickle.dump(vocab, output)
    output.close()

def create_dataset(opt, SRC, TRG, TRG_REL):
    """
    This function is used to construct the training batches from the input. 
    opt: parameters 
    SRC: source data for machine translation. 
    TRG: target data for machine translation. 
    TRG_REL: retrieval data for machine translation. This data is constructed apriori using TRG sentences as 
    queries to the retrieval corpus and retrieving a passage from the most relevant document. we call it relevance 
    data 
    """    
    print("creating dataset and iterator... ")
    src_data = [line.strip() for line in opt.src_data]
    trg_data = [line for line in opt.trg_data] 
    rm_data = [line for line in open(opt.rm_data)] 
    raw_data = {'src' : src_data, 'trg': trg_data, 'trg_rel': rm_data} 
    
    trg_data_retrieval = [line for line in opt.trg_data_retrieval]
    raw_data_retrieval = {'trg': trg_data_retrieval} 
    
    #using the retrieval data to compute idf of every token in the retrieval corpus 
    vectorizer = TfidfVectorizer(use_idf=True)
    vectorizer.fit_transform(raw_data['trg'] + raw_data_retrieval['trg'])
    tokens = vectorizer.get_feature_names()
    idf_values = vectorizer.idf_
    opt.idf_dict = {}
    for i in range(len(tokens)):
        opt.idf_dict.setdefault(tokens[i],idf_values[i]) 
    
    #chopping off sentences with maximum string length parameter
    df = pd.DataFrame(raw_data, columns=["src", "trg", "trg_rel"])
    mask = (df['src'].str.count(' ') < opt.max_strlen) & (df['trg'].str.count(' ') < opt.max_strlen)
    df = df.loc[mask]
    df.to_csv("translate_transformer_temp.csv", index=False)
    data_fields = [('src', SRC), ('trg', TRG) , ('trg_rel', TRG_REL)]
    train = data.TabularDataset('./translate_transformer_temp.csv', format='csv', fields=data_fields)
    train_iter = MyIterator(train, batch_size=opt.batchsize, device=opt.device,
                        repeat=False, sort_key=lambda x: (len(x.src), len(x.trg)),
                        batch_size_fn=batch_size_fn, train=True, shuffle=True)
    os.remove('translate_transformer_temp.csv')  
    
    
    opt.src_pad = SRC.vocab.stoi['<pad>']
    opt.trg_pad = TRG.vocab.stoi['<pad>']    
    opt.train_len = get_len(train_iter)

    return train_iter

opt.device = 0 if opt.no_cuda is False else -1
if opt.device == 0:
    assert torch.cuda.is_available()
print(opt.device)
read_data(opt)

SRC, TRG, TRG_REL = create_fields(opt)
keyboard()
opt.train = create_dataset(opt, SRC, TRG, TRG_REL)
opt.valid = create_dataset(opt, SRC, TRG, TRG_REL)

#TRG = create_retrieval_vocabulary(opt, TRG)

if opt.checkpoint > 0:
    print("model weights will be saved every %d minutes and at end of epoch to directory weights/"%(opt.checkpoint))
    

CONTEXT_SIZE = 2            
model = get_model(opt, len(SRC.vocab), len(TRG.vocab), CONTEXT_SIZE)
opt.optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr, betas=(0.9, 0.98), eps=1e-9)
opt.optimizer_wemb = torch.optim.Adam(model.parameters(), lr = 0.000001, betas=(0.9, 0.98), eps=1e-9)
if opt.SGDR == True:
    opt.sched = CosineWithRestarts(opt.optimizer, T_max=opt.train_len)
    
print(TRG.vocab.itos[0])
print(len(TRG.vocab))
opt.idf_dict["<sos>"] = 1
opt.idf_dict["<eos>"] = 1
class_weights = []
for i in range(len(TRG.vocab)):
    if TRG.vocab.itos[i] in opt.idf_dict:
        class_weights.append(opt.idf_dict[TRG.vocab.itos[i]])
    else:
        class_weights.append(0.0001)
class_weights = torch.FloatTensor(class_weights)

from torch import nn 
import subprocess
opt.checkpoint = 5
nll_loss = nn.NLLLoss() # loss function
            
def train_model(model, opt, SRC, TRG):
    # keyboard()()
    torch.cuda.empty_cache()
    best_loss = 0.00
    print("training model...")
    opt.idf_dict["<sos>"] = 1
    opt.idf_dict["<eos>"] = 1
    
    #class weight indicates the weight of each term in calculating the loss function 
    #we have set it as the idf of the terms. 
    class_weights = []
    for i in range(len(TRG.vocab)):
        if TRG.vocab.itos[i] in opt.idf_dict:
            class_weights.append(opt.idf_dict[TRG.vocab.itos[i]])            
        else:
            class_weights.append(0.0001)
    
    class_weights = torch.FloatTensor(class_weights)
    model.train()
    class_weights = class_weights.cuda()
    
    start = time.time()
    #If we are checkpointing model after a certain period of time, we will use cptime to keep track of time
    if opt.checkpoint > 0:
        cptime = time.time()
        checkpointing_step = 0
        
    #the variable to keep track of global step 
    #print("number of batches: ".format(len(opt.train)))
    
    for epoch in range(opt.epochs):
        print("beginning epoch: {}".format(epoch))
        # keyboard()()
        step = 0
        translation_loss = 0
        embedding_loss = 0
                
        for i, batch in enumerate(opt.train):
            #in each batch we have three inputs src, trg and trg_retrieval. This is the part where it is different 
            #from traditional machine translation. trg_retrieval comes from the retrieval corpus. please refer to the 
            #paper that for each sentence in trg we retrieve from a corpus and add it to the parallel data. 
            torch.cuda.empty_cache()
            start = time.clock()
            # src -> torch.Size([27, 36]) #  trg -> torch.Size([27, 34]) # trg_rel -> torch.Size([27, 202])
            src = batch.src.transpose(0,1)
            trg = batch.trg.transpose(0,1)    
            trg_rel = batch.trg_rel.transpose(0,1) 
            #the length of trg is pretty small and that's why we are adding it three times and finally adding it with 
            #trg_rel. trg_rel comes from the retrieval corpus. This is why and how we construct tt. 
            tt = torch.cat((trg[: , :-1], trg[: , :-1], trg[: , :-1], trg_rel), 1).numpy()
            
            # tt.shape -> (27,301)
            
            trg_input = trg[:, :-1]
            src_mask, trg_mask = create_masks(src, trg_input, opt)            
            src_mask = src_mask.cuda()            
            trg_mask = trg_mask.cuda()
            src = src.cuda() 
            trg_input = trg_input.cuda()  

            data_context_final = []
            data_target_final = []
            #now creating the training batches for word embedding on the fly. 
            for element in tt: 
                # keyboard()()
                #removing the index of some unnecessay tokens
                indices = [i for i, x in enumerate(element) if x == 0 or x == 1 or x == 2 or x ==3]
                element = np.delete(element, indices)
                np.random.shuffle(element)
                corpus_text = torch.tensor(element)
                #creating word embedding data. The pivot will be at index i. 
                for i in range(CONTEXT_SIZE, len(corpus_text) - CONTEXT_SIZE):
                    #at first create the context 
                    data_context = []
                    data_target = []
                    for j in range(CONTEXT_SIZE):
                        data_context.append(corpus_text[i - CONTEXT_SIZE + j])
                    for j in range(1, CONTEXT_SIZE + 1):
                        data_context.append(corpus_text[i + j])
                    #now create the pivot or the target 
                    data_target.append(corpus_text[i])
                    #add context and pivot to create a batch
                    data_context_final.append(torch.LongTensor(data_context))
                    data_target_final.append(torch.LongTensor(data_target))
                    
            #Now we are training word embedding. 
            keyboard()
            offset = 1000
            number_of_batches = int(len(data_context_final) / offset) # 11 
            index = 1
            loss_wemb = 0
            for batch in range(number_of_batches):
                opt.optimizer_wemb.zero_grad()
                #creating a batch of 1000 context, target pairs 
                data_context_final_temp = torch.stack(data_context_final[index:index+offset]).cuda()
                data_target_final_temp = torch.stack(data_target_final[index:index+offset]).cuda()            
                preds_emb = model(src, trg_input, src_mask, trg_mask, data_context_final_temp)
                #computing word embedding loss for 1000 data points for the word embedding task  
                loss_wemb_temp = F.cross_entropy(preds_emb.view(-1, preds_emb.size(-1)), data_target_final_temp.contiguous().view(-1).cuda())
                #make word embedding loss less impactful 
                loss_wemb_temp/= 10
                loss_wemb+=loss_wemb_temp.item()
                index+=1
                loss_wemb_temp.backward()
                opt.optimizer_wemb.step()
                
            #Now using the model to translate src to trg_input. 
            preds = model(src, trg_input, src_mask, trg_mask, 2)
            #The original translations 
            ys = trg[:, 1:].contiguous().view(-1).cuda()
            opt.optimizer.zero_grad()
            #computing cross-entropy loss for the translation task 
            loss = F.cross_entropy(preds.view(-1, preds.size(-1)), ys, weight=class_weights, ignore_index=opt.trg_pad)              
            loss.backward()

            translation_loss+= loss.item()
            embedding_loss+= loss_wemb/index

            opt.optimizer.step()
            
            if opt.SGDR == True: 
                opt.sched.step()
            
            step+=1
            
            
            #reading the evaluation from the IR task. specifically focusing on mAP. 
            # p = subprocess.Popen('CUDA_VISIBLE_DEVICES=3 python translate_validation.py', shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            # line = p.stdout.readlines()[0]
            # mAP = float(line.decode("utf-8").strip().split()[1])
        preds = model(src, trg_input, src_mask, trg_mask, 2)
        
        if loss < best_loss:
            best_best_loss = loss
            torch.save(model.state_dict(), 'weights/model_weights_best_validation')
            # cptime = time.time()
            # info = { 'loss': loss.item(), 'map': mAP}
            # for tag, value in info.items():
            #     logger.scalar_summary(tag, value, checkpointing_step+1)
            # checkpointing_step+=1
            
            # print(str(opt.epoch) + "\tepoch loss nmt\t" + str(translation_loss/step) + "\tepoch loss embedding\t" + str(embedding_loss/step) + "\tepoch info")                
            # torch.save(model.state_dict(), 'weights/model_weights_' + str(epoch))

model = model.cuda()
train_model(model, opt, SRC, TRG)