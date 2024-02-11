from transformers import AutoTokenizer
import torch
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import copy 

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class CreateDataset:
  """
    This class will tokenize and create a dataset for 1-2 heads models
  
  """
  def __init__ (self,sentences,labels1,labels2,model_name,batch_size=32,max_length=128):
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
                                             padding='max_length',
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



class Create3HEADDataset:
  """
    Same with `CreateDataset` but for 3 heads models
  """
  def __init__ (self,sentences,labels1,labels2,labels3,model_name,batch_size=32,max_length=128):
    self.tokenizer=AutoTokenizer.from_pretrained(model_name, use_fast=False)
    self.batch_size=batch_size
    self.model_name=model_name
    self.sentences=np.array(sentences)
    self.labels1 = labels1
    self.labels2 = labels2
    self.labels3 = labels3
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
                                             padding='max_length',
                                             truncation = True,
                                             return_attention_mask=True,
                                             return_tensors='pt')
        input_ids.append(encoded_dict['input_ids'])
        attention_masks.append(encoded_dict['attention_mask'])
    self.input_ids = torch.cat(input_ids,dim=0).to(DEVICE)
    self.attention_masks = torch.cat(attention_masks,dim=0).to(DEVICE)
    self.labels1 = torch.tensor(self.labels1).type(torch.LongTensor).to(DEVICE)
    self.labels2 = torch.tensor(self.labels2).type(torch.LongTensor).to(DEVICE)
    self.labels3 = torch.tensor(self.labels3).type(torch.LongTensor).to(DEVICE)
    self.sent_index = torch.tensor(sent_index).to(DEVICE)

  def todataloader(self):
    self.encoder_generator()


    self.dataset = TensorDataset(self.input_ids, self.attention_masks ,self.labels1,self.labels2,self.labels3)
    # generator = torch.Generator(device=DEVICE)
    self.data_loader = DataLoader(self.dataset,
                                  # sampler=RandomSampler(self.dataset),
                                  batch_size=self.batch_size,
                                  shuffle=True
                                  # generator = generator,
                                )
    return self.data_loader

  
# Since we cannot use the `DataCollatorForLanguageModeling` from the `transformers` library 
# while training a customized model, we need to create our own `DataCollator` class called 
# `DataCollatorHandMade`

# It will have a method called `random_label` that will randomly mask some tokens in the input, 
# followed by the traditional rule for masking tokens:

# - 30% of the tokens will be masked follow the rule:

# - 80% of the time, replace the token with the `[MASK]` token
# - 10% of the time, keep the token unchanged
# - 10% of the time, replace the token with a random token
class DataCollatorHandMade:
    
    def __init__(self,model_name, mlm_prob = 0.3):
      self.tokenizer = AutoTokenizer.from_pretrained(model_name)
      self.mask_token_id = self.tokenizer.mask_token_id
      self.mlm_prob = mlm_prob
  
    def random_label(self,input_ids: torch.Tensor,attention_mask: torch.Tensor):
        mlm_inputs = []
        labels =[]
        total_mask = 0
        
        for input_id, att in zip(input_ids,attention_mask):
          mlm_input = input_id.clone()
          max_pos  = int(torch.sum(att))
          
          num_mask = int(max_pos * self.mlm_prob) 
          total_mask += num_mask      
          mask_pos = torch.randint(0, max_pos, size=(num_mask,), dtype=torch.int32)
          
          mask = torch.zeros(len(mlm_input))
          mask[mask_pos] = 1
          
          mask = mask.type(torch.bool)
          
          mlm_input[mask] = self.mask_token_id
          label = copy.deepcopy(input_id)
          label[mlm_input != self.mask_token_id]=-100

          
          lucky_mask = torch.randint(1,11,size = (num_mask,), dtype=torch.int32)
          # This 8/1/1 code looks messy. It's just a way to randomly replace the masked tokens
          
          for i,lucky in enumerate(lucky_mask):
            if int(lucky)%8==0: # 10%
              mlm_input[mask_pos[i]] = input_id[mask_pos[i]]
            elif int(lucky)%7==0: # 10%
              mlm_input[mask_pos[i]] = torch.randint(0,64000,size=(1,))[0]
          
          mlm_inputs.append(mlm_input.reshape(1,-1))
          labels.append(label.reshape(1,-1))
          
        mlm_inputs = torch.cat(mlm_inputs, dim=0).to(DEVICE)
        labels = torch.cat(labels, dim=0).to(DEVICE)
        return mlm_inputs, labels, total_mask

# To calculate the loss for the masked language model, we need to create a function called 
# `label_for_mlm`. This function will take the result from the model and the random masked labels, 
# and return the predicted values and the labels in format which compatible with nn.CrossEntroyLoss.
      
def label_for_mlm(result, mlm_labels):
    y_pred=[]
    labels=[]
    for i in range(result.shape[0]):
        tmp_res = torch.zeros(result.shape[1])
        tmp_res[mlm_labels[i]!=-100] = 1
        
        for j in range(result.shape[1]):
            if tmp_res[j]==1:
                lb = mlm_labels[i,j]
                pred = result[i][j]
                y_pred.append(pred.reshape(1,-1))
                labels.append(lb)

    labels= torch.Tensor(labels).type(torch.LongTensor).to(DEVICE)
    y_pred = torch.cat(y_pred,dim=0).to(DEVICE)  
    
    return y_pred, labels
      
      
        
      
  
      

   