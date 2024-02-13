import torch
from Trainer.mlm_4head_trainer import Trainer
from utils.dataloader import Create4HEADDataset
import pandas as pd

# import wandb
# wandb.login(key="b46a760f71842e87d8ac966f77b2db06d0a7085a")

architectures=["linear"]
bert_name="vinai/phobert-base-v2"


train_set = pd.read_csv('dataset/victsd_train2.csv')
test_set  = pd.read_csv('dataset/victsd_test2.csv')
is_smart = True
percentages = [0.5]
for architecture in architectures:
  for p in percentages :
    extract = False
  
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("___________",bert_name,"____________")
    # wandb.init(
    #   project = "2-Head_Bert",
    #   name = bert_name + architecture + "2-head_vfsc" + "smart",
    # )
    batch_size = 32
    # epochs =40
    # if extract:
    epochs = 50
    # if "large" in model_name:
    #   batch_size=4
    # varient='Epoch-17-2-head-linear-smart-5_5-3head'
    # mlm_data_loader = CreateMLMDataset(mlm_set['text'][:train_set.shape[0]*2].values,bert_name, batch_size=batch_size*2 ).todataloader()
    train_data_loader = Create4HEADDataset(train_set['text'], train_set['label_x'],train_set['label_y'],train_set['label_z'], bert_name, batch_size=batch_size).todataloader()
    test_data_loader  = Create4HEADDataset(test_set['text'], test_set['label_x'],test_set['label_y'],test_set['label_z'], bert_name, batch_size=batch_size).todataloader()
    bertcnn=Trainer(bert_name,  train_data_loader, test_data_loader, model=architecture,is_smart=is_smart,extract=extract,varient='Epoch-6-2-head-linear-smart-5_5-3head-vsfc-redo')
    bertcnn.fit(schedule=True,epochs=epochs,report=False,name=f"{architecture}-victsd",percentage= p)
    # wandb.finish()

    del bertcnn
    del train_data_loader
    del test_data_loader
    torch.cuda.empty_cache()

    print("_______________End__________________")