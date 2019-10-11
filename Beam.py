import torch
from Batch import nopeak_mask
import torch.nn.functional as F
import math
import numpy as np


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


def k_best_outputs(outputs, out, log_scores, i, k, TRG):
    probs, ix = out[:, -1].data.topk(k)

    preds_token_ids = ix.view(ix.size(0), -1)  # size = k * k
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
            if prob > 0.001:
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
    query_tokens = []
    for i in range(2, opt.max_len):

        trg_mask = nopeak_mask(i, opt)
        src_mask = src_mask.cuda()
        trg_mask = trg_mask.cuda()

        out = model.out(model.decoder(outputs[:, :i], e_outputs, src_mask, trg_mask))
        # print (outputs.size())
        # print (out.size())
        out = F.softmax(out, dim=-1)

        # print("output data shape")
        # print(out.data.shape)

        outputs, log_scores, pred_strings, pred_strings_dict = k_best_outputs(outputs, out, log_scores, i, opt.k, TRG)

        #         This part is another way of forming the query dictionary
        for pred_string in pred_strings:
            pred_string_splitted = pred_string.split()
            for st in pred_string_splitted:
                query.setdefault(st, 1.0)
                query[st] = query[st] + 1
            query_tokens.extend(pred_string_splitted)

        for term in pred_strings_dict:
            if term in query:
                if pred_strings_dict[term] > query[term]:
                    query[term] = pred_strings_dict[term]
            else:
                query[term] = pred_strings_dict[term]

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
        return query_list, query, query_tokens

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
        return query_list, query, query_tokens
