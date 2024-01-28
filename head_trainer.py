import wandb
import torch
import torch.nn as nn
from bert2head.model import BertCNN2HEAD, BertLinear2HEAD, BertCNN2HEAD_UIT
from loss_function import SMARTLoss , kl_loss, sym_kl_loss
from torch.optim import lr_scheduler,AdamW, Adam
import numpy as np
from tqdm import tqdm
import time
from pcg import PCGrad

from sklearn.metrics import accuracy_score,f1_score

class Trainer:
  def __init__(self,name,train_data_loader,test_data_loader,model="cnn",lora=False,is_smart=True,extract=False):

    self.extract = extract
    self.is_smart = is_smart
    self.model_type = model
    self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    self.model_name = name
    self.lora=lora
    self.tokenizer=None
    
    if model == "cnn":

      self.bertcnn=BertCNN2HEAD(name)
    elif model == "cnn-uit":
      self.bertcnn=BertCNN2HEAD_UIT(name)
  

    else:
      self.bertcnn=BertLinear2HEAD(name)
      
    if extract:
      if is_smart:
        self.bertcnn.BertModel.load_state_dict(torch.load("models/finetuneBertsmart.pt"))
      else:
        self.bertcnn.BertModel.load_state_dict(torch.load("models/finetuneBert.pt"))
    self.bertcnn=self.bertcnn.to(self.device)
    self.train_data_loader = train_data_loader
    self.test_data_loader  = test_data_loader

    self.is_schedule = False
    self.model_prameters = list(self.bertcnn.parameters())
    self.optimizer = AdamW(self.model_prameters, lr=2e-5, eps=5e-9)
    self.criterion = nn.CrossEntropyLoss().to(self.device)
    self.smart_loss_fn = SMARTLoss(eval_fn = self.bertcnn, loss_fn = kl_loss, loss_last_fn = sym_kl_loss)
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
      b_clas = batch[3].to(self.device)
      self.optimizer.zero_grad()

      sent_predictions, clas_predictions = self.bertcnn(b_input_ids,b_input_mask)

      # if self.is_smart:
      loss1 = self.criterion(sent_predictions, b_sent) + self.weight * self.smart_loss_fn(b_input_ids, sent_predictions, b_input_mask,sent=True)
      loss2 = self.criterion(clas_predictions, b_clas) + self.weight * self.smart_loss_fn(b_input_ids, clas_predictions, b_input_mask,sent=False)
      # else:
      # loss1 = self.criterion(sent_predictions, b_sent)
      # loss2 = self.criterion(clas_predictions, b_clas)
        
 
      t_loss = loss1*self.percentage + loss2*(1-self.percentage)
      
      

      acc_sent = self.categorical_accuracy(sent_predictions, b_sent)
      acc_clas = self.categorical_accuracy(clas_predictions, b_clas)

      self.optimizer.zero_grad()
      t_loss.backward()
      self.optimizer.step()

      epoch_loss += t_loss.item()
      epoch_acc_sent += acc_sent.item()
      epoch_acc_clas += acc_clas.item()
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
        b_clas = batch[3].to(self.device)

        sent_predictions, clas_predictions = self.bertcnn(b_input_ids,b_input_mask)
        loss1 = self.criterion(sent_predictions, b_sent) 
        loss2 = self.criterion(clas_predictions, b_clas) 

        t_loss = (0.7*loss1 + loss2*0.3)*2
        epoch_loss += t_loss.item()

        sent_predictions = sent_predictions.detach().cpu().numpy()
        clas_predictions = clas_predictions.detach().cpu().numpy()

        label_sent = b_sent.to('cpu').numpy()
        label_clas = b_clas.to('cpu').numpy()

        pred1, true1 = self.predictions_labels(sent_predictions,label_sent)
        pred2, true2 = self.predictions_labels(clas_predictions,label_clas)

        all_pred_sent.extend(pred1)
        all_pred_clas.extend(pred2)

        all_true_sent.extend(true1)
        all_true_clas.extend(true2)

    val_accuracy_sent = accuracy_score(all_pred_sent,all_true_sent)
    val_accuracy_clas = accuracy_score(all_pred_clas,all_true_clas)

    sent_f1_score = f1_score(all_pred_sent,all_true_sent,average='macro')
    sent_f1_scorew = f1_score(all_pred_sent,all_true_sent,average='weighted')
    clas_f1_score = f1_score(all_pred_clas,all_true_clas,average='macro')
    clas_f1_scorew = f1_score(all_pred_clas,all_true_clas,average='weighted')

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
      temp = 15
    # self.scheduler = lr_scheduler.LinearLR(self.optimizer, start_factor=1.0, end_factor=0.16667/2, total_iters=epochs)
    self.scheduler = lr_scheduler.LinearLR(self.optimizer, start_factor=1.0, end_factor=(0.16667/epochs)*temp, total_iters=epochs)
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
      
      with open(f"log\log-{self.name}{'-smart'if self.is_smart else ''}-{'boosting-' if self.extract else ''}{int(self.percentage*10)}_{10-int(self.percentage*10)}-v1.2.txt","a") as f:
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

        if valid_accs[1]>=0.8 and valid_accs[0]>=0.843 and valid_f1s[0]>=0.85 and valid_f1s[1]>=0.72:
            torch.save(self.bertcnn.state_dict(),f'models/{self.model_type}/Epoch-{epoch+1}-2-head-{self.model_type}-{"smart"if self.is_smart else ""}{"-boosting" if self.extract else ""}-{int(self.percentage*10)}_{10-int(self.percentage*10)}-pcg.pt')
            f.write(f'\nModel epoch {epoch+1} saved')
        f.write('\n=============Epoch Ended==============')
        print('\n=============Epoch Ended==============')