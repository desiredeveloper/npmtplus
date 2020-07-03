# -*- coding: utf-8 -*-
"""coding np2mt.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/19hkwsAu2f0mvQOCrwoT7FHZKJK7-V_-W
"""

import torch
import argparse
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from torchtext.datasets import TranslationDataset
from torchtext.data import Field, BucketIterator
from model import Encoder, TargetEncoder, Attention, Decoder

import spacy
import numpy as np
import sys
import random
import math
import time

parser = argparse.ArgumentParser()

parser.add_argument(
    "--seed", default=1234, help="Initialize with random SEED", type=int)
parser.add_argument(
    "--embed", default=128, help="Embedding dimension", type=int)
parser.add_argument(
    "--hidden", default=128, help="Hidden dimension", type=int)
parser.add_argument(
    "--segment", default=128, help="Segment RNN dimension", type=int)
parser.add_argument(
    "--layers", default=2, help="Number of layers", type=int)
parser.add_argument(
    "--lr", default=0.001, help="Learning Rate", type=float)
parser.add_argument(
    "--dropout", default=0.4, help="Dropout value", type=float)
parser.add_argument(
    "--segmentThresold", default=4, help="Maximum length for a segment", type=int)
parser.add_argument(
    "--maxLen", default=40, help="Maximum decoding length", type=int)
parser.add_argument(
    "--data", default='data/IITB_instance', help="Path of dataset")
parser.add_argument(
    "--source", default='.en', help="Source language (extension of data files)")
parser.add_argument(
    "--target", default='.hi', help="Target language (extension of data files)")
parser.add_argument(
    "--clip", default=1, help="Clip gradient value", type=float)
parser.add_argument(
    "--epochs", default=12, help="Number of epochs", type=int)
parser.add_argument(
    "--dictionary", default=True, help="Should use dictionary", type=bool)
parser.add_argument(
    "--batch", default=8, help="Batch Size", type=int)
parser.add_argument(
    "--saveTo", default='npmt-model.pt', help="save model to path", type=str)
parser.add_argument(
    "--loadFrom", default='npmt-model.pt', help="load model from path", type=str)

args = parser.parse_args(sys.argv[1:])

SEED = args.seed
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.autograd.set_detect_anomaly(True)

embed_dim = args.embed
hidden_dim = args.hidden
segment_dim = args.segment
n_layers = args.layers
dropout = args.dropout
segment_threshold = args.segmentThresold
MAX_LENGTH = args.maxLen

spacy_en = spacy.load('en')
writer = SummaryWriter()

def tokenize_en(text):
    """
    Tokenizes English text from a string into a list of strings (tokens) and reverses it
    """
    return [tok.text for tok in spacy_en.tokenizer(text)]

def tokenize_hi(text):
    """
    Tokenizes Hindi text from a string into a list of strings (tokens) 
    """
    return text.split()

SRC = Field(tokenize = tokenize_en, 
            init_token = '<sos>', 
            eos_token = '<eos>', 
            lower = True)

TRG = Field(tokenize = tokenize_hi, 
            init_token = '<sos>', 
            eos_token = '<eos>', 
            lower = True)

train_data, valid_data, test_data  = TranslationDataset.splits(
                                      path=args.data,
                                      validation='dev',
                                      exts = (args.source, args.target), 
                                      fields = (SRC, TRG))

print(f"Number of training examples: {len(train_data.examples)}")
print(f"Number of validation examples: {len(valid_data.examples)}")
print(f"Number of testing examples: {len(test_data.examples)}")

vars(train_data.examples[0])

SRC.build_vocab(train_data, min_freq = 1)
TRG.build_vocab(train_data, min_freq = 1,specials=['<pad>','<sop>','<eop>'])

print(f"Unique tokens in source (en) vocabulary: {len(SRC.vocab)}")
print(f"Unique tokens in target (hi) vocabulary: {len(TRG.vocab)}")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

device 

BATCH_SIZE = args.batch

train_iterator, valid_iterator, test_iterator = BucketIterator.splits(
    (train_data, valid_data, test_data),
    batch_size = BATCH_SIZE, 
    device = device)

"""# EnCoder Parameters"""

input_dim = len(SRC.vocab)
output_dim = len(TRG.vocab)

class NP2MT(nn.Module):
  def __init__(self, encoder, attention, targetEncoder, decoder, device):
    super().__init__()
    
    self.encoder = encoder
    self.attention = attention
    self.targetEncoder = targetEncoder
    self.decoder = decoder
    self.device = device
      
  def forward(self, src, trg, teacher_forcing_ratio = 0.5):
    
    #src = [src len, batch size]
    #trg = [trg len, batch size]
    #teacher_forcing_ratio is probability to use teacher forcing
    #e.g. if teacher_forcing_ratio is 0.75 we use teacher forcing 75% of the time
    
    batch_size = src.shape[1]
    trg_len = trg.shape[0]
    trg_vocab_size = self.decoder.output_dim
    
    #encoder_outputs is representation of all phrases states of the input sequence, back and forwards
    #hidden is the final forward and backward hidden states, passed through a linear layer (batch_size*hidden_dim)
    encoder_outputs, hidden = self.encoder(src)
    output_target_decoder,hidden_target_decoder = self.targetEncoder(trg)
    sop_symbol = TRG.vocab.stoi['<sop>']
    eop_symbol = TRG.vocab.stoi['<eop>']
    
    alpha = torch.zeros(batch_size,trg_len).to(self.device)
    # alpha[:,0] = 1
    for end in range(1,trg_len):

      x_i = torch.zeros(batch_size,min(end,segment_threshold)).to(self.device)
      index = 0
      for phraseLen in range(min(end,segment_threshold),0,-1):
        start = end - phraseLen + 1
        weighted, attn = self.attention(encoder_outputs, output_target_decoder[start-1])
        
        sop_vector = (torch.ones(1,batch_size,dtype=torch.int64)*sop_symbol).to(self.device)
        input_phrase = trg[start:end+1,:]
        input_phrase = torch.cat((sop_vector,input_phrase),0)
        eop_vector = (torch.ones(1,batch_size,dtype=torch.int64)*eop_symbol).to(self.device)
        input_phrase = torch.cat((input_phrase,eop_vector),0)
        
        phraseProb = torch.zeros(batch_size).to(self.device)
        
        for t in range(input_phrase.shape[0]-1):

          probabilities = self.decoder(input_phrase[t].view(-1,batch_size), weighted)
          phraseProb += probabilities[torch.arange(batch_size),input_phrase[t+1]]
          
        x_i[:,index] = alpha[:,start-1] + phraseProb
        index = index+1
        
      alpha[:,end] = torch.logsumexp(x_i,dim=1)
      
    outFinal = alpha[:,-1]
    del x_i
    del alpha
    del phraseProb
    return outFinal
  
  def attendedPhrase(self, attn, src_len):
    
    X = torch.zeros((src_len,src_len)).to(self.device)
    X[torch.triu(torch.ones(src_len, src_len)) == 1] = attn
    maxIndex = X.argmax()
    i = maxIndex/src_len
    j = maxIndex%src_len
    # i = src_len - 2 - int(sqrt(-8*k + 4*src_len*(src_len-1)-7)/2.0 - 0.5)
    # j = k + i + 1 - src_len*(src_len-1)/2 + (src_len-i)*((src_len-i)-1)/2
    return i, j
    
  def buildPhrase(self, src, start, end):
    outString = ""
    for i in range(start,end+1):
      outString += SRC.vocab.itos[src[i]]
      outString += " "
    return outString[:-1]
    
  def forward_predict(self, src, reference, dictionary):
      
    batch_size = src.shape[1]

    sos_symbol = TRG.vocab.stoi['<sos>']    
    sop_symbol = TRG.vocab.stoi['<sop>']
    eop_symbol = TRG.vocab.stoi['<eop>']
    eos_symbol = TRG.vocab.stoi['<eos>']
    unk_symbol = TRG.vocab.stoi['<unk>']
    #encoder_outputs is representation of all phrases states of the input sequence, back and forwards
    #hidden is the final forward and backward hidden states, passed through a linear layer (batch_size*hidden_dim)
    
    for b in range(batch_size):
      encoder_outputs, hidden = self.encoder(src[:,b].view(-1,1))
      trg = torch.tensor([[sos_symbol]]).to(self.device)
      output_target_decoder,hidden_target_decoder = self.targetEncoder(trg)
      weighted, attn = self.attention(encoder_outputs, output_target_decoder[-1])
      num_segments = 0
      decoded_words = []

      avlDictionary = False
      while len(decoded_words)<= MAX_LENGTH:
        new_segment = False
        
        decoder_input = torch.tensor([[sop_symbol]]).to(self.device)
        for j in range(segment_threshold):
          probabilities = self.decoder(decoder_input.view(-1,1), weighted)
          prob, decoder_output = torch.max(probabilities,1)
          if decoder_output == eop_symbol or decoder_output == eos_symbol:
            break
          else:
            if not new_segment:
              num_segments = num_segments + 1
              new_segment = True
              curr_segment = []
          
          decoder_input = decoder_output
          curr_segment.append(decoder_output)
        
        src_start, src_end = self.attendedPhrase(attn, src.shape[0])
        outPhrase = self.buildPhrase(src[:,b],src_start,src_end)
        if unk_symbol in curr_segment and dictionary.get(outPhrase) is not None:
          dictTarget = dictionary.get(outPhrase)
          li = list(dictTarget.split(" "))
          indexli = map(lambda x: TRG.vocab.stoi(x),li)
          decoded_words.extend(indexli)
        else:
          decoded_words.extend(curr_segment)

        if decoder_output == eos_symbol:
          break
        if len(curr_segment) > 0: # if the very first sop gives eop
          trg = torch.cat(decoded_words).view(-1,1)
        output_target_decoder,hidden_target_decoder = self.targetEncoder(trg)
        weighted, attn = self.attention(encoder_outputs, output_target_decoder[-1])
        
        
      # print("Source Input")
      outString = ""
      for i in src[:,b]:
        outString += SRC.vocab.itos[i]
        outString += " "
      with open('predictions/src.txt', 'a') as f:
        print(outString,file=f)
              
      # print("Reference output")
      outString = ""
      for i in reference[:,b]:
        outString += TRG.vocab.itos[i]
        outString += " "
      with open('predictions/ref.txt', 'a') as f:
        print(outString,file=f)

      # print("model's output")
      outString = ""
      for i in decoded_words:
        outString += TRG.vocab.itos[i]
        outString += " "
      with open('predictions/sys.txt', 'a') as f:
        print(outString,file=f)

  
attn = Attention(hidden_dim, hidden_dim)
enc = Encoder(input_dim, embed_dim, hidden_dim, segment_dim, n_layers, dropout, segment_threshold, device)
targetEnc = TargetEncoder(output_dim, embed_dim, hidden_dim, segment_dim, n_layers, dropout)
dec = Decoder(output_dim, embed_dim, hidden_dim, segment_dim, n_layers, dropout)

model = NP2MT(enc, attn, targetEnc, dec, device).to(device)

def init_weights(m):
  for name, param in m.named_parameters():
    if 'weight' in name:
      nn.init.normal_(param.data, mean=0, std=0.01)
    else:
      nn.init.constant_(param.data, 0)

model.apply(init_weights)

def count_parameters(model):
  return sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f'The model has {count_parameters(model):,} trainable parameters')

optimizer = optim.Adam(model.parameters(),lr=args.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)

def train(model, iterator, optimizer, clip, epoch):
  
  model.train()
  epoch_loss = 0
  for i, batch in enumerate(iterator):
    src = batch.src
    trg = batch.trg
    optimizer.zero_grad()
    output = model(src, trg)
    
    loss = -output.mean()
    writer.add_scalar("Loss/train", loss, epoch)
    print("loss is",loss)
    loss.backward()
    epoch_loss += loss.item()
    torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
    optimizer.step()
    epoch_loss += loss.item()
    del loss
    del output
    # print(torch.cuda.memory_cached()//(1024*1024))
    torch.cuda.empty_cache()
    
  return epoch_loss / len(iterator)

def evaluate(model, iterator, epoch):
    
  model.eval()
  epoch_loss = 0
  
  with torch.no_grad():
    for i, batch in enumerate(iterator):
        src = batch.src
        trg = batch.trg
        output = model(src, trg) #turn off teacher forcing
        loss = -output.mean()
        writer.add_scalar("Loss/valid", loss, epoch)
        epoch_loss += loss.item()
        del loss
        del output
        torch.cuda.empty_cache()
  return epoch_loss / len(iterator)


def predict(model, iterator, dictionary):
    
  model.eval()
  
  with torch.no_grad():
    for i, batch in enumerate(iterator):
        src = batch.src
        trg = batch.trg
        model.forward_predict(src, trg, dictionary) #turn off teacher forcing

def epoch_time(start_time, end_time):
  elapsed_time = end_time - start_time
  elapsed_mins = int(elapsed_time / 60)
  elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
  return elapsed_mins, elapsed_secs

N_EPOCHS = args.epochs
CLIP = args.clip

best_valid_loss = float('inf')

for epoch in range(N_EPOCHS):
    
  start_time = time.time()
  
  train_loss = train(model, train_iterator, optimizer, CLIP, epoch)
  writer.flush()
  valid_loss = evaluate(model, valid_iterator, epoch)
  writer.flush()
  
  end_time = time.time()
  
  epoch_mins, epoch_secs = epoch_time(start_time, end_time)
  
  if valid_loss < best_valid_loss:
    best_valid_loss = valid_loss
    torch.save(model.state_dict(), 'checkpoints/npmt-epoch{}.pt'.format(epoch+1))
  
  print(f'Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s')
  print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
  print(f'\t Val. Loss: {valid_loss:.3f} |  Val. PPL: {math.exp(valid_loss):7.3f}')

writer.close()

model.load_state_dict(torch.load(args.loadFrom))

if args.dictionary:
  import csv
  with open('dict.csv', mode='r') as infile:
    reader = csv.reader(infile,delimiter='\t')
    mydict = {rows[0]:rows[1] for rows in reader}
  predict(model, test_iterator, mydict)

