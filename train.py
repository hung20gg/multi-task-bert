import torch
from head_trainer import Trainer
from dataloader import CreateDataset
import pandas as pd

import wandb
wandb.login(key="b46a760f71842e87d8ac966f77b2db06d0a7085a")

architectures=["linear"]
bert_name="vinai/phobert-base-v2"


train_set = pd.read_csv('dataset/Train/train2.csv')
test_set  = pd.read_csv('dataset/Test/test2.csv')
is_smart = True
percentages = [0.5,0.7]
for architecture in architectures:
  for p in percentages :
    extract = False
    # if extract and architecture in ["linear"]:
    #   continue
    # if not extract and architecture in ["cnn",'cnn-uit']:
    #   continue
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("___________",bert_name,"____________")
    wandb.init(
      project = "2-Head_Bert",
      name = bert_name + architecture + "2-head_vfsc" + "smart",
    )
    batch_size = 32
    # epochs =40
    # if extract:
    epochs = 40
    # if "large" in model_name:
    #   batch_size=4

    train_data_loader = CreateDataset(train_set['text'], train_set['label_x'],train_set['label_y'], bert_name, batch_size=batch_size).todataloader()
    test_data_loader  = CreateDataset(test_set['text'], test_set['label_x'],test_set['label_y'], bert_name, batch_size=batch_size).todataloader()
    bertcnn=Trainer(bert_name,  train_data_loader, test_data_loader, model=architecture,is_smart=is_smart,extract=extract)
    bertcnn.fit(schedule=True,epochs=epochs,report=True,name=f"{architecture}-pcg",percentage= p)
    wandb.finish()

    del bertcnn
    del train_data_loader
    del test_data_loader
    torch.cuda.empty_cache()

    print("_______________End__________________")