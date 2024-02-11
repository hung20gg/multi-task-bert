import torch
from torch.nn import Module 
import torch.nn as nn
from transformers import AutoModel

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

"""
    BERT x heads model with a MLM head
    
    After passing tokenize input into BERT, the output will be passing 2 different 
    layers:
    
    - MLM layer:  1 Linear(embedibng_dim, embedding_dim) 
                  1 LayerNorm(embedding_dim)
                  1 Linear(embedding_dim, max_vocab_size)
        The architecture is similar to `AutoModelForMaskLM` on `huggingface`
      
    - Classificationn layer:  1 Linear(embedding_dim, embedding_dim)
                              1 LayerNorm(embedding_dim)  
                              Here the output will be divided into 2 heads:
                              1 Linear(embedding_dim, 4) for sentence classification
                              1 Linear(embedding_dim, 10) for topic classification
    
    Will more tasks, users can add more heads to the classification layers.
"""

# @title Finetune Model

class Linear3HEAD(Module):
    def __init__(self, embedding_dim, dropout=0.2,activation=None):

        super().__init__()
        self.ln1= nn.LayerNorm(embedding_dim)
        self.ln2= nn.LayerNorm(embedding_dim)
        # self.ln2= nn.LayerNorm(len(filter_sizes)*n_filters)
        self.fc_input = nn.Linear(embedding_dim, embedding_dim)
        self.fc_input2 = nn.Linear(embedding_dim, embedding_dim)
        
        self.activation = nn.SiLU() if activation is None else activation


        self.out_sent = nn.Linear(embedding_dim,4)
        self.out_clas = nn.Linear(embedding_dim,10)
        self.MLM = nn.Linear(embedding_dim,64001)

        self.dropout_sent = nn.Dropout(dropout)
        self.dropout_clas = nn.Dropout(dropout)
    def forward(self, encoded,encoded2=None, mlm=False):
        
        embedded = self.activation(self.fc_input(encoded)) # bs, 64, 768

        # Warmup truoc khi chia head
        embedded = self.ln1(embedded)
       
        sent = self.out_sent(embedded)
        clas = self.out_clas(embedded)
        if mlm:
          embedded2 = self.activation(self.fc_input2(encoded2))
          mlm = self.MLM(embedded2)
          return sent, clas, mlm

        return sent, clas


class BertLinear3HEAD(Module):
    def __init__(self, name):
        super().__init__()
        self.BertModel = AutoModel.from_pretrained(name)
        self.BertModel.to(device)
        # self.phoBertModel.load_state_dict(torch.load(pretrained_path))
        self.linear = Linear3HEAD(768)
    def forward(self,sentences,attention,sentences2=None, mlm=False):
       embedded = self.BertModel(sentences,attention_mask=attention).last_hidden_state[:,0,:]
       embedded2 = nn.Identity(768)
       if mlm:
         embedded2 = self.BertModel(sentences2,attention)[0]
       return self.linear(embedded,embedded2, mlm=mlm)
       
       
