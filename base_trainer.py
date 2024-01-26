from base_model.model import BaseModel
import torch
import pandas as pd

print(torch.cuda.is_available())
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

s_train = pd.read_csv('dataset/Train/clas_train2.csv')
s_test = pd.read_csv('dataset/Test/clas_test2.csv')

model=BaseModel("vinai/phobert-large")
model.fit(s_train['text'],s_train['label'],evaluate=(s_test['text'],s_test['label']),batch_size=32,epochs=40,eval_steps=500,save_name='v2.1')