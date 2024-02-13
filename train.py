import torch
from Trainer.mlm_3head_trainer import Trainer
from utils.dataloader import CreateDataset
import pandas as pd

# If you want to use wandb, uncomment the lines that are commented

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
    # If `extract` = True, the model will be loaded from a checkpoint, and you have to pass
    # the checkpoint path to the `varient` parameter in Trainer
    extract = False
    

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("___________",bert_name,"____________")
    # wandb.init(
    #   project = "2-Head_Bert",
    #   name = bert_name + architecture + "2-head_vfsc" + "smart",
    # )
    batch_size = 32
  
    epochs = 40
    # if "large" in model_name:
    #   batch_size=4
    
    # varient='Epoch-17-2-head-linear-smart-5_5-3head'
    train_data_loader = CreateDataset(train_set['text'], train_set['label_x'],train_set['label_y'], bert_name, batch_size=batch_size).todataloader()
    test_data_loader  = CreateDataset(test_set['text'], test_set['label_x'],test_set['label_y'], bert_name, batch_size=batch_size).todataloader()
    bertcnn=Trainer(bert_name,  train_data_loader, test_data_loader, model=architecture,is_smart=is_smart,extract=extract,varient='Epoch-6-2-head-linear-smart-5_5-3head-vsfc-redo')
    bertcnn.fit(schedule=True,epochs=epochs,report=False,name=f"{architecture}-victsd",percentage= p)
    # wandb.finish()

    del bertcnn
    del train_data_loader
    del test_data_loader
    torch.cuda.empty_cache()

    print("_______________End__________________")