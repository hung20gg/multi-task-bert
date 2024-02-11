import wandb
import torch
import torch.nn as nn
from architecture.bert2head_mlm.model import  BertLinear1HEADMLM
from utils.loss_function import SMARTLoss1Label , kl_loss, sym_kl_loss
from torch.optim import lr_scheduler,AdamW
import numpy as np
from tqdm import tqdm
import time
from dataloader import DataCollatorHandMade, label_for_mlm

from sklearn.metrics import accuracy_score,f1_score

class Trainer:
  def __init__(self,name,train_data_loader,test_data_loader,model="cnn",lora=False,is_smart=True,extract=False,varient='hehe'):

    self.extract = extract
    self.is_smart = is_smart
    self.model_type = model
    self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    self.model_name = name
    self.lora=lora
    self.tokenizer=None
    self.datacollator = DataCollatorHandMade(self.model_name)
    
    
    self.bertcnn=BertLinear1HEADMLM(name)
      
    if extract:
      # if is_smart:
      #   self.bertcnn.BertModel.load_state_dict(torch.load("models/finetuneBertsmart.pt"))
      # else:
        self.bertcnn.load_state_dict(torch.load(f"models/{model}/{varient}.pt"))
    self.bertcnn=self.bertcnn.to(self.device)
    self.train_data_loader = train_data_loader
    # self.mlm_data_loader = mlm_data_loader
    self.test_data_loader  = test_data_loader

    self.is_schedule = False
    self.model_prameters = list(self.bertcnn.parameters())
    self.optimizer = AdamW(self.model_prameters, lr=1.8e-5, eps=5e-9)
    self.criterion = nn.CrossEntropyLoss().to(self.device)
    self.smart_loss_fn = SMARTLoss1Label(eval_fn = self.bertcnn, loss_fn = kl_loss, loss_last_fn = sym_kl_loss)
    self.weight = 0.02

  def categorical_accuracy(self,preds, y):
    max_preds = preds.argmax(dim = 1, keepdim = True) # get the index of the max probability
    correct = max_preds.squeeze(1).eq(y)
    return correct.sum() / torch.FloatTensor([y.shape[0]]).to(self.device)

  def predictions_labels(self,preds,labels):
    pred = np.argmax(preds,axis=1).flatten()
    label = labels.flatten()
    return pred,label

  def train(self):
    epoch_loss = 0
    epoch_acc_sent = 0
    epoch_acc_clas = 0
    self.bertcnn.train()
    for batch in tqdm(self.train_data_loader):
      b_input_ids = batch[0].to(self.device)
      b_input_mask = batch[1].to(self.device)
      b_sent = batch[2].to(self.device)
           
      mlm_input_ids, mlm_labels, total_mask = self.datacollator.random_label(b_input_ids,b_input_mask)   
  
      self.optimizer.zero_grad()

      # sent_predictions, clas_predictions = self.bertcnn(b_input_ids, b_input_mask)
      sent_predictions, mlm_predictions = self.bertcnn(b_input_ids, b_input_mask, mlm_input_ids, mlm=True)
      mlm_predictions, mlm_labels = label_for_mlm(mlm_predictions, mlm_labels)
      
      # if self.is_smart:
      loss1 = self.criterion(sent_predictions, b_sent) + self.weight * self.smart_loss_fn(b_input_ids, sent_predictions, b_input_mask)
      
      loss3 = self.criterion(mlm_predictions, mlm_labels)
      # # else:
      # loss1 = self.criterion(sent_predictions, b_sent)
      # loss2 = self.criterion(clas_predictions, b_clas)
      avg_mask = total_mask/mlm_input_ids.shape[0]
 
      # t_loss = loss1 + loss2 
      t_loss = loss1 +  loss3/avg_mask
      

      acc_sent = self.categorical_accuracy(sent_predictions, b_sent)


      self.optimizer.zero_grad()
      t_loss.backward()
      self.optimizer.step()

      epoch_loss += t_loss.item()
      epoch_acc_sent += acc_sent.item()
      
    if self.is_schedule:
      self.scheduler.step()
    return epoch_loss / len(self.train_data_loader), epoch_acc_sent / len(self.train_data_loader), epoch_acc_clas / len(self.train_data_loader)
  
  def eval(self):
    epoch_loss = 0
    all_true_sent = []
    all_true_clas = []
    all_pred_sent = []
    all_pred_clas = []

    self.bertcnn.eval()

    with torch.no_grad():

      for batch in tqdm(self.test_data_loader):
        b_input_ids = batch[0].to(self.device)
        b_input_mask = batch[1].to(self.device)
        b_sent = batch[2].to(self.device)

        sent_predictions = self.bertcnn(b_input_ids,b_input_mask)
        loss1 = self.criterion(sent_predictions, b_sent) 
        
        t_loss = loss1
        epoch_loss += t_loss.item()

        sent_predictions = sent_predictions.detach().cpu().numpy()

        label_sent = b_sent.to('cpu').numpy()
        pred1, true1 = self.predictions_labels(sent_predictions,label_sent)

        all_pred_sent.extend(pred1)

        all_true_sent.extend(true1)

    val_accuracy_sent = accuracy_score(all_pred_sent,all_true_sent)
    val_accuracy_clas = 0

    sent_f1_score = f1_score(all_pred_sent,all_true_sent,average='macro')
    sent_f1_scorew = f1_score(all_pred_sent,all_true_sent,average='weighted')
    clas_f1_score = 0
    clas_f1_scorew = 0

    accs = (val_accuracy_sent,val_accuracy_clas)
    f1s= (sent_f1_score,clas_f1_score,sent_f1_scorew,clas_f1_scorew)

    avg_val_loss = epoch_loss/len(self.test_data_loader)
    return avg_val_loss, accs,  f1s

  def epoch_time(self,start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

  def fit(self,schedule=True,epochs=20,optim=None,report = False,name="saved",percentage=0.7):
    self.percentage = percentage
    temp = 30
    if self.extract:
      temp = 5
    # self.scheduler = lr_scheduler.LinearLR(self.optimizer, start_factor=1.0, end_factor=0.16667/2, total_iters=epochs)
    self.scheduler = lr_scheduler.LinearLR(self.optimizer, start_factor=1.0, end_factor=(0.2/epochs)*temp, total_iters=epochs)
    self.name=name
    if optim!=None:
      self.optimizer=optim
    self.is_schedule=schedule
    
    
    for epoch in range(epochs):
      start_time = time.time()
      train_loss, train_acc_sent, train_acc_clas = self.train()
      valid_loss, valid_accs, valid_f1s = self.eval()
      end_time = time.time()
 

      if report:
        wandb.log({'Train loss':train_loss,
                   'Validation loss':valid_loss,
                   'Train accuracy sent':train_acc_sent,
                   'Train accuracy clas':train_acc_clas,
                   'Val acc sent':valid_accs[0],
                   'F1m score sent':valid_f1s[0],
                   'F1w score sent':valid_f1s[0],
                   'Val acc clas':valid_accs[1],
                   'F1m score clas':valid_f1s[1],
                   'F1w score clas':valid_f1s[3],
                   })

      epoch_mins, epoch_secs = self.epoch_time(start_time, end_time)
      
      with open(f"log\log-{self.name}{'-smart'if self.is_smart else ''}-{'boosting-' if self.extract else ''}{int(self.percentage*10)}_{10-int(self.percentage*10)}-3head-vihsd-redo.txt","a") as f:
        f.write(f'\nEpoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')

        f.write(f'\n\tTrain Loss : {train_loss:.3f}  | Val. Loss  : {valid_loss:.3f}')
        f.write(f'\n\tTrn.acc se : {train_acc_sent*100:.2f}% | Trn.acc ca : {train_acc_clas*100:.2f}%')
        f.write(f'\n\tVal.acc se : {valid_accs[0]*100:.2f}% | Val.acc ca : {valid_accs[1]*100:.2f}%')
        f.write(f'\n\tVal.F1m se : {valid_f1s[0]*100:.2f}  | Val.F1m ca : {valid_f1s[1]*100:.2f}')
        f.write(f'\n\tVal.F1w se : {valid_f1s[2]*100:.2f}  | Val.F1w ca : {valid_f1s[3]*100:.2f}')
        
        print(f'Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')

        print(f'\tTrain Loss : {train_loss:.3f}  | Val. Loss  : {valid_loss:.3f}')
        print(f'\tTrn.acc se : {train_acc_sent*100:.2f}% | Trn.acc ca : {train_acc_clas*100:.2f}%')
        print(f'\tVal.acc se : {valid_accs[0]*100:.2f}% | Val.acc ca : {valid_accs[1]*100:.2f}%')
        print(f'\tVal.F1m se : {valid_f1s[0]*100:.2f}  | Val.F1m ca : {valid_f1s[1]*100:.2f}')
        print(f'\tVal.F1w se : {valid_f1s[2]*100:.2f}  | Val.F1w ca : {valid_f1s[3]*100:.2f}')

        if  valid_accs[0]>=0.875 and valid_f1s[0]>=675 :
            torch.save(self.bertcnn.state_dict(),f'models/{self.model_type}/Epoch-{epoch+1}-2-head-{self.model_type}-{"smart"if self.is_smart else ""}{"-boosting" if self.extract else ""}-{int(self.percentage*10)}_{10-int(self.percentage*10)}-3head-vihsd-redo.pt')
            f.write(f'\nModel epoch {epoch+1} saved')
        f.write('\n=============Epoch Ended==============')
        print('\n=============Epoch Ended==============')