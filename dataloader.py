from transformers import AutoTokenizer
import torch
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class CreateDataset:
  def __init__ (self,sentences,labels1,labels2,model_name,batch_size=32,max_length=64):
    self.tokenizer=AutoTokenizer.from_pretrained(model_name, use_fast=False)
    self.batch_size=batch_size
    self.model_name=model_name
    self.sentences=np.array(sentences)
    self.labels1 = labels1
    self.labels2 = labels2
    self.max_length=max_length
    self.device=  torch.device('cuda' if torch.cuda.is_available() else 'cpu')


  def encoder_generator(self):
    # sentences= self.sentences
    
    
    sent_index = []
    input_ids = []
    attention_masks =[]
    for index,sent in enumerate(self.sentences):
        sent_index.append(index)
        encoded_dict = self.tokenizer.encode_plus(sent,
                                             add_special_tokens=True,
                                             max_length=self.max_length,
                                             pad_to_max_length=True,
                                             truncation = True,
                                             return_attention_mask=True,
                                             return_tensors='pt')
        input_ids.append(encoded_dict['input_ids'])
        attention_masks.append(encoded_dict['attention_mask'])
    self.input_ids = torch.cat(input_ids,dim=0).to(DEVICE)
    self.attention_masks = torch.cat(attention_masks,dim=0).to(DEVICE)
    self.labels1 = torch.tensor(self.labels1).type(torch.LongTensor).to(DEVICE)
    self.labels2 = torch.tensor(self.labels2).type(torch.LongTensor).to(DEVICE)
    self.sent_index = torch.tensor(sent_index).to(DEVICE)

  def todataloader(self):
    self.encoder_generator()


    self.dataset = TensorDataset(self.input_ids, self.attention_masks ,self.labels1,self.labels2)
    # generator = torch.Generator(device=DEVICE)
    self.data_loader = DataLoader(self.dataset,
                                  # sampler=RandomSampler(self.dataset),
                                  batch_size=self.batch_size,
                                  shuffle=True
                                  # generator = generator,
                                )
    return self.data_loader