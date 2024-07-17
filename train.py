import torch
# from trainer.mlm_head_trainer import Trainer
from trainer.base_trainer import Trainer
from utils.dataloader import CreateDataset
import pandas as pd
import gc

# If you want to use wandb, uncomment the lines that are commented

# import wandb
# wandb.login(key="b46a760f71842e87d8ac966f77b2db06d0a7085a")

architectures=["linear"]

bert_names=["vinai/phobert-large", "xlm-roberta-large", "uitnlp/CafeBERT"]
bert_names=["vinai/phobert-base-v2", "xlm-roberta-base",'uitnlp/visobert']



is_smart = True
tasks = ['classification']

for task in tasks:
  for bert_name in bert_names :
    extract = False
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    print("___________",bert_name,"____________")

    batch_size = 128
    # epochs =40
    # if extract:
    epochs = 15
    

    if 'vinai' in bert_name:
      train_set = pd.read_csv('dataset/train_set_processed.csv')
      test_set  = pd.read_csv('dataset/test_set_processed.csv')
      val_set = pd.read_csv('dataset/val_set_processed.csv')
    else:
        train_set = pd.read_csv('dataset/train_set.csv')
        test_set  = pd.read_csv('dataset/test_set.csv')
        val_set = pd.read_csv('dataset/val_set.csv')
        
    train_data_loader = CreateDataset(train_set['text'], train_set['sentiment'],train_set['classification'], bert_name, batch_size=batch_size).todataloader()
    test_data_loader  = CreateDataset(test_set['text'], test_set['sentiment'],test_set['classification'], bert_name, batch_size=batch_size).todataloader()
    val_data_loader  = CreateDataset(val_set['text'], val_set['sentiment'],val_set['classification'], bert_name, batch_size=batch_size).todataloader()
    # bertcnn=Trainer(bert_name,  train_data_loader, test_data_loader, model=architecture,is_smart=is_smart,extract=extract,varient='Epoch-6-2-head-linear-smart-5_5-3head-vsfc-redo')
    trainer = Trainer(bert_name, task)
    
    bert_name = bert_name.split('/')[-1]
    trainer.train(train_data_loader, val_data_loader, epochs=15, save_name=f"{bert_name}-{task}")
    acc, f1m, f1w = trainer.evaluate(test_data_loader, save_name=f"{bert_name}-{task}")
    print(f'Final Prediction\n Model :{bert_name} | {task}\nAcc: {acc}, f1m: {f1m}, f1w: {f1w}')

    del trainer
    del train_data_loader
    del test_data_loader
    gc.collect()
    torch.cuda.empty_cache()

    print("_______________End__________________")