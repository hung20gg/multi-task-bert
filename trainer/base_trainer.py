import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification,TrainingArguments, Trainer, DataCollatorWithPadding
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
import copy
from tqdm import tqdm
class Trainer:
    def __init__(self, model_name, task):
        self.model_name = model_name
        # self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.task = task
        if task == 'sentiment':
            num_labels = 4
        elif task == 'topic':
            num_labels = 10
            
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.loss_fn = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=5e-5)
        
    def train(self, train_dataloader, val_dataloader, epochs=10, save_name='model'):
    
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(self.optimizer, max_lr=0.001, steps_per_epoch=len(train_dataloader), epochs=epochs)
        
        final_model = None
        f1m = 0
        acc = 0
        f1w = 0
        
        for epoch in range(epochs):
            self.model.train()
            for batch in tqdm(train_dataloader):
                self.optimizer.zero_grad()
                input_ids = batch[0].to(self.device)
                attention_mask = batch[1].to(self.device)
                if self.task == 'sentiment':
                    labels = batch[2].to(self.device)
                else:
                    labels = batch[3].to(self.device)
                    
                outputs = self.model(input_ids, attention_mask=attention_mask)
                
                loss = self.loss_fn(outputs.logits, labels)
                loss.backward()
                self.optimizer.step()
                self.scheduler.step()
            
            acc, f1m, f1w = self.evaluate(val_dataloader)
            print(f'Epoch {epoch+1}/{epochs}, acc: {acc}, f1m: {f1m}, f1w: {f1w}')
            if f1m > f1m:
                final_model.save_pretrained(f'models/linear/{save_name}.pt')
            
    def evaluate(self, dataloader, save_name=''):
        if save_name != '':
            self.model.load_state_dict(torch.load(f'models/linear/{save_name}.pt'))
        self.model.eval()
        all_preds = []
        all_labels = []
        for batch in dataloader:
            input_ids = batch[0].to(self.device)
            attention_mask = batch[1].to(self.device)
            if self.task == 'sentiment':
                labels = batch[2].to(self.device)
            else:
                labels = batch[3].to(self.device)
            with torch.no_grad():
                outputs = self.model(input_ids, attention_mask=attention_mask)
                logits = outputs.logits
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            
        acc = accuracy_score(all_labels, all_preds)
        f1m = f1_score(all_labels, all_preds, average='macro')
        f1w = f1_score(all_labels, all_preds, average='weighted')
        return acc, f1m, f1w