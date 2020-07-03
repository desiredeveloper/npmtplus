import torch
import torch.nn as nn
import torch.nn.functional as F

"""# Building Encoder"""

class Encoder(nn.Module):
  def __init__(self,input_dim,embed_dim,hidden_dim,segment_dim,n_layers,dropout,segment_threshold,device):
    super().__init__()
    self.input_dim = input_dim
    self.hidden_dim = hidden_dim
    self.n_layers = n_layers
    self.segment_threshold = segment_threshold
    self.segment_dim = segment_dim
    self.device = device
    
    self.embedding = nn.Embedding(input_dim,embed_dim)
    self.rnn = nn.GRU(embed_dim,hidden_dim,n_layers,dropout=dropout,bidirectional=True)

    self.segmentRnn = nn.GRU(hidden_dim*2,segment_dim,n_layers,dropout=dropout)
    # self.fc = nn.Linear(hidden_dim*2,hidden_dim)
    self.dropout = nn.Dropout(dropout)

  def forward(self,input):

    #input = [src len, batch size]
    embedded = self.dropout(self.embedding(input))
    #embedded = [src len, batch size, emb dim]

    outputs, hidden = self.rnn(embedded)
    #outputs = [src len, batch size, hid dim * num directions]
    #hidden = [n layers * num directions, batch size, hid dim]
        
    segment_encoding, hidden = self.segment_rnn(outputs)
    #segment_encoding = [src len* (src len+1)/2, batch size, segment_dim*num_directions]
    #hidden = [n layers * num_directions, batch size, hid dim]

    # hidden = torch.tanh(self.fc(torch.cat((hidden[-2],hidden[-1]),dim=1)))

    return segment_encoding,hidden

  def segment_rnn(self,outputs):
    N = outputs.shape[0]
    batch_size = outputs.shape[1]
    dp_forward = torch.zeros(N, N, batch_size, self.segment_dim).to(self.device)
    dp_backward = torch.zeros(N, N, batch_size, self.segment_dim).to(self.device)

    for i in range(N):
      hidden_forward = torch.randn(self.n_layers, batch_size, self.hidden_dim).to(self.device)
      for j in range(i, min(N, i + self.segment_threshold)):
        
        # outputs[j] = [batch size, hidden_dim* num_direction]
        next_input = outputs[j].unsqueeze(0)
        # next_input = [1, batch size, hidden_dim* num_direction]
        
        out, hidden_forward = self.segmentRnn(next_input,hidden_forward)
        #out = [1, batch size, segment_dim]
        #hidden_forward = [n layers , batch size, hid dim]

        dp_forward[i][j] = out.squeeze(0)

    for i in range(N):
      hidden_backward = torch.randn(self.n_layers, batch_size, self.hidden_dim).to(self.device)
      for j in range(i, max(-1, i - self.segment_threshold), -1):

        # outputs[j] = [batch size, hidden_dim* num_direction]
        next_input = outputs[j].unsqueeze(0)
        # next_input = [1, batch size, hidden_dim* num_direction]
        
        out, hidden_backward = self.segmentRnn(next_input,hidden_backward)
        #out = [1, batch size, segment_dim]
        #hidden_backward = [n layers , batch size, hid dim]
        
        dp_backward[j][i] = out.squeeze(0)
    
    dp = torch.cat((dp_forward,dp_backward),dim=3)
    dp = dp[torch.triu(torch.ones(N, N)) == 1]
    return dp,torch.cat((hidden_forward,hidden_backward),dim=2)

"""# Defining Attn Network"""
'''
Attention is calculated over encoder_outputs S(i,j) and context representation
of previously generated segments (from Target Decoder)

'''
class Attention(nn.Module):
  def __init__(self, enc_hid_dim, dec_hid_dim):
    super().__init__()

    self.attn = nn.Linear((enc_hid_dim * 2) + dec_hid_dim, dec_hid_dim)
    self.v = nn.Linear(dec_hid_dim, 1, bias = False)

  def forward(self, encoder_outputs, output_target_decoder):
      
    #encoder_outputs = [no. of segments, batch size, enc hid dim * 2]
    #output_target_decoder = [batch size, dec hid dim]
    batch_size = encoder_outputs.shape[1]
    src_len = encoder_outputs.shape[0]
    
    #repeat decoder hidden state src_len times
    output_target_decoder = output_target_decoder.unsqueeze(1).repeat(1, src_len, 1)
    
    encoder_outputs = encoder_outputs.permute(1, 0, 2)
    
    #output_target_decoder = [batch size, no. of segments, dec hid dim]
    #encoder_outputs = [batch size, no. of segments, enc hid dim * 2]
    
    energy = torch.tanh(self.attn(torch.cat((output_target_decoder, encoder_outputs), dim = 2))) 
    #energy = [batch size,  no. of segments, dec hid dim]
    attention = self.v(energy).squeeze(2)
    #attention= [batch size,  no. of segments]
    a = F.softmax(attention, dim=1)
    #a = [batch size,  no. of segments]
    a = a.unsqueeze(1)
    #a = [batch size, 1,  no. of segments]
    weighted = torch.bmm(a, encoder_outputs)
    #weighted = [batch size, 1, enc hid dim * 2]
    weighted = weighted.permute(1, 0, 2)
    #weighted = [1, batch size, enc hid dim * 2]
    return weighted, attention

"""# Building Decoder"""


class TargetEncoder(nn.Module):
  def __init__(self, output_dim, embed_dim, hidden_dim,segment_dim,n_layers, dropout):
    super().__init__()
    self.output_dim = output_dim
    self.n_layers = n_layers
    self.embedding = nn.Embedding(self.output_dim, embed_dim)
    self.rnn = nn.GRU(embed_dim,hidden_dim,n_layers,dropout=dropout)
    self.dropout = nn.Dropout(dropout)

  def forward(self, input):
          
    #input = [target_len,batch size]
    #hidden = [batch size, dec hid dim]
    #encoder_outputs = [src len, batch size, enc hid dim * 2]
    
    embedded = self.dropout(self.embedding(input))
    #embedded = [target_len, batch size, emb dim]
    
    output_target_decoder,hidden_target_decoder = self.rnn(embedded)
    #output_target_decoder = [target_len, batch size, hidden_dim]
    #hidden_target_decoder = [n layers , batch size, hidden_dim]
    return output_target_decoder,hidden_target_decoder
  

class Decoder(nn.Module):
  def __init__(self, output_dim, embed_dim, hidden_dim,segment_dim,n_layers, dropout):
    super().__init__()
    self.output_dim = output_dim
    self.n_layers = n_layers
    self.hidden_dim = hidden_dim
    self.embedding = nn.Embedding(self.output_dim, embed_dim)
    self.segmentRnn = nn.GRU(hidden_dim,hidden_dim,n_layers,dropout=dropout)
    self.fc_out = nn.Linear((hidden_dim * 2) + hidden_dim + embed_dim, self.output_dim)
    self.soft = nn.LogSoftmax(dim=1)
    self.dropout = nn.Dropout(dropout)
  
  def forward(self, input, weighted):
    
    embedded = self.dropout(self.embedding(input))
    output, hidden = self.segmentRnn(embedded)
    
    output = output.squeeze(0)
    weighted = weighted.squeeze(0)
    embedded = embedded.squeeze(0)
    
    prediction = self.soft(self.fc_out(torch.cat((output, weighted, embedded), dim = 1)))
    #prediction = [batch size, output dim]
    # probabilities = self.soft(prediction)
    # phraseProb *= torch.exp(probabilities[torch.arange(batch_size),input[t+1]])
    
    return prediction