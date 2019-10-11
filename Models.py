import torch
import torch.nn as nn 
from Layers import EncoderLayer, DecoderLayer
from Embed import Embedder, PositionalEncoder
from Sublayers import Norm
import torch.nn.functional as F
import copy

def get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

class Encoder(nn.Module):
    def __init__(self, vocab_size, d_model, N, heads, dropout):
        super().__init__()
        self.N = N
        self.embed = Embedder(vocab_size, d_model)
        self.pe = PositionalEncoder(d_model, dropout=dropout)
        self.layers = get_clones(EncoderLayer(d_model, heads, dropout), N)
        self.norm = Norm(d_model)
    def forward(self, src, mask):
        x = self.embed(src)
        x = self.pe(x)
        for i in range(self.N):
            x = self.layers[i](x, mask)
        return self.norm(x)
    
class Decoder(nn.Module):
    def __init__(self, vocab_size, d_model, N, heads, dropout):
        super().__init__()
        self.N = N
        self.embed = Embedder(vocab_size, d_model)
        self.pe = PositionalEncoder(d_model, dropout=dropout)
        self.layers = get_clones(DecoderLayer(d_model, heads, dropout), N)
        self.norm = Norm(d_model)
    def forward(self, trg, e_outputs, src_mask, trg_mask):
        x = self.embed(trg)
        x = self.pe(x)
        for i in range(self.N):
            x = self.layers[i](x, e_outputs, src_mask, trg_mask)
        return self.norm(x)

class Transformer(nn.Module):
    def __init__(self, src_vocab, trg_vocab, d_model, N, heads, dropout, context_size):
        super().__init__()
        self.encoder = Encoder(src_vocab, d_model, N, heads, dropout)
        self.decoder = Decoder(trg_vocab, d_model, N, heads, dropout)
        self.out = nn.Linear(d_model, trg_vocab)  #d_model is the embedding size and trg_vocab is the english vocabulary size

        self.vocab_size = trg_vocab
        self.embedding_size = d_model
        self.context_size = context_size
        # return vector size will be context_size*2*embedding_size
        #if self.context_size != 0:
        self.lin1 = nn.Linear(self.context_size * 2 * self.embedding_size, d_model)
        #self.lin2 = nn.Linear(512, self.vocab_size)


    # def forward(self, src, trg, src_mask, trg_mask):
    #     e_outputs = self.encoder(src, src_mask)
    #     #print("DECODER")
    #     d_output = self.decoder(trg, e_outputs, src_mask, trg_mask)
    #     output = self.out(d_output)
    #     return output

    # def forward(self, src, trg, src_mask, trg_mask, context):
    #     e_outputs = self.encoder(src, src_mask)
    #     #print("DECODER")
    #     d_output = self.decoder(trg, e_outputs, src_mask, trg_mask)
    #     output = self.out(d_output)
    #
    #     if type(context) == int:
    #         return output
    #     else:
    #         dim = context.size(0)
    #         word_embedding_out = self.decoder.embed(context).view(dim , -1)
    #         #word_embedding_out = word_embedding_out.view(1, -1)
    #         word_embedding_out = self.lin1(word_embedding_out)
    #         word_embedding_out = F.relu(word_embedding_out)
    #         word_embedding_out = self.out(word_embedding_out)
    #         word_embedding_out = F.log_softmax(word_embedding_out, dim=0)
    #
    #         return output, word_embedding_out

    def forward(self, src, trg, src_mask, trg_mask, context):

        if type(context) == int:
            e_outputs = self.encoder(src, src_mask)
            # print("DECODER")
            d_output = self.decoder(trg, e_outputs, src_mask, trg_mask)
            output = self.out(d_output)
            return output
        else:
            dim = context.size(0)
            word_embedding_out = self.decoder.embed(context).view(dim , -1)
            #word_embedding_out = word_embedding_out.view(1, -1)
            word_embedding_out = self.lin1(word_embedding_out)
            word_embedding_out = F.relu(word_embedding_out)
            word_embedding_out = self.out(word_embedding_out)
            word_embedding_out = F.log_softmax(word_embedding_out, dim=0)

            return word_embedding_out


def get_model(opt, src_vocab, trg_vocab, context_size):
    
    assert opt.d_model % opt.heads == 0
    assert opt.dropout < 1

    model = Transformer(src_vocab, trg_vocab, opt.d_model, opt.n_layers, opt.heads, opt.dropout, context_size)
       
    if opt.load_weights is not None:
        print("loading pretrained weights...")
        model.load_state_dict(torch.load(f'{opt.load_weights}/model_weights'))
    else:
        for p in model.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p) 
    
    if opt.device == 0:
        model = model.cuda()
    
    return model
    
