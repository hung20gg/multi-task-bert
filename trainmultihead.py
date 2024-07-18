import torch
# from trainer.mlm_head_trainer import Trainer
from trainer.head_trainer import Trainer
from utils.dataloader import CreateDataset
import pandas as pd
import gc

# If you want to use wandb, uncomment the lines that are commented

# import wandb
# wandb.login(key="b46a760f71842e87d8ac966f77b2db06d0a7085a")

architectures=["linear"]

# bert_names=["google-bert/bert-base-multilingual-uncased", "FPTAI/velectra-base-discriminator-cased"]
bert_names=['uitnlp/visobert']



is_smart = False
tasks = ['sentiment', 'classification']

for bert_name in bert_names :
    extract = False
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    print("___________",bert_name,"____________")

    batch_size = 128
    # epochs =40
    # if extract:
    epochs = 20
    

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
    trainer=Trainer(bert_name,  train_data_loader, val_data_loader,is_smart=is_smart,extract=extract)
    bert_name = bert_name.split('/')[-1]
    trainer.fit(epochs=epochs,report=False,name=f"{bert_name}")
    # Note: I forgot to +1 in epochs, so the model will train in 19 epochs instead of 20 epochs
    del trainer
    del train_data_loader
    del test_data_loader
    gc.collect()
    torch.cuda.empty_cache()

    print("_______________End__________________")