# @title Auto automodel

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification,TrainingArguments, Trainer, DataCollatorWithPadding
import evaluate
import numpy as np

class Dataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels=None):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        if self.labels:
            item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.encodings["input_ids"])

class BaseModel:
  def __init__(self, name,save_name="",task="sent"):
    self.task = task
    self.name=name
    self.tokenizer=AutoTokenizer.from_pretrained(self.name, use_fast=False,ignore_mismatched_sizes=True)
    self.model = None
    self.max_length=64
    self.vnemolex=False
    self.train_dataset=None
    self.test_dataset=None
    self.metric = evaluate.load("accuracy")
    self.metric2 = evaluate.load("f1")
    self.epochs=5
    self.eval_steps=500
    self.evaluation_strategy="steps"
    self.training_args=None
    self.output_dir="test_trainer"
    self.batch_size = 32
    self.seed = 42
    self.trainer = None
    self.save_name = save_name
    x = torch.tensor([1, 2, 3])
    if torch.cuda.is_available():
      device = torch.device("cuda")
      x = x.to(device)
      print("Tensor moved to GPU")
    else:
      print("GPU is not available, using CPU instead")


  def tokenize(self, data,max_length=64):
    self.max_length=max_length
    return self.tokenizer(data, padding=True, truncation=True, max_length=self.max_length)

  def compute_metrics(self,eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    acc =  self.metric.compute(predictions=predictions, references=labels)["accuracy"]
    f1  =  self.metric2.compute(predictions=predictions, references=labels, average="macro")["f1"]
    return {"accuracy": acc, "f1": f1}


  def dataset_no_split(self,data1,label1,data2,label2):
    X_train=list(data1)
    y_train=list(label1)
    X_test=list(data2)
    y_test=list(label2)
    self.labels=len(set(y_train))

    X_train = self.tokenizer(X_train, padding=True, truncation=True, max_length=self.max_length)
    X_test  = self.tokenizer(X_test,  padding=True, truncation=True, max_length=self.max_length)
    
    
    self.model = AutoModelForSequenceClassification.from_pretrained(self.name, num_labels=self.labels,ignore_mismatched_sizes=True)
    self.train_dataset = Dataset(X_train, y_train)
    self.test_dataset = Dataset(X_test, y_test)





  def trainarg(self, eval_steps=500,evaluation_strategy="epoch",epochs=50,output_dir="test_trainer",batch_size=16,**kwargs):
    self.batch_size=batch_size
    self.output_dir=output_dir
    self.epochs=epochs
    self.eval_steps=eval_steps
    self.evaluation_strategy=evaluation_strategy
    self.training_args=TrainingArguments(output_dir=self.output_dir,
                                  evaluation_strategy=self.evaluation_strategy,
                                    eval_steps=self.eval_steps,
                                    per_device_train_batch_size=self.batch_size,
                                    per_device_eval_batch_size=self.batch_size,
                                    num_train_epochs=self.epochs,
                                    save_strategy = "epoch",
                                         weight_decay=0.01,
                                    metric_for_best_model='f1',
                                    seed=self.seed,
                                    save_total_limit = 1,
                                    load_best_model_at_end=True,
                                        #  report_to="wandb"
                                         )

  def train_model(self):
    self.device  = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    self.model = self.model.to(self.device)
    self.trainer = Trainer(
      model=self.model,
      args=self.training_args,
      train_dataset=self.train_dataset,
      eval_dataset=self.test_dataset,
      compute_metrics=self.compute_metrics,
      data_collator = DataCollatorWithPadding(tokenizer = self.tokenizer),
      # callbacks=[EarlyStoppingCallback(early_stopping_patience=5)],


  )
    return self.trainer


  def fit(self,data,label,evaluate=None,eval_steps=500,evaluation_strategy='epoch'
          ,epochs=5,output_dir="test_trainer",batch_size=16,max_length=64
          ,save_name='',**kwargs):



    self.dataset_no_split(data,label,evaluate[0],evaluate[1])
   
    # if self.task != "sent":
    #   wandb.init(
    #     project = "Neu_clas_v4.3",
    #     name = self.name+self.save_name
    # )
    # else:
    #   wandb.init(
    #       project = "Neu_sen_v4.3",
    #       name = self.name+self.save_name
    #   )
    self.trainarg(eval_steps,evaluation_strategy,epochs,output_dir,batch_size)
    train=self.train_model()
    train.train()
    train.save_model(f'models/{self.name}_{self.task}_{save_name}')
    # wandb.finish()
  def eva(self):
    self.trainer.evaluate()
  def save(self,target=None):
    if not target:
      target="fine_tune_"+self.name
    self.trainer.save_model(target)
