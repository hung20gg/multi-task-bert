# from dataloader import CreateDataset
# from bert3head.model import BertLinear3HEAD
# import torch

# model = BertLinear3HEAD("vinai/phobert-base-v2")
# model.load_state_dict(torch.load('models/linear/3head-boosting2.pt'))
# model = model.cuda()

# def remove_number(text):
#     new_text = ''
#     for char in text:
#         if not char.isdigit():
#             new_text += char
#     return new_text

# import numpy as np
# import pandas as pd
# from tqdm import tqdm

# df = pd.read_csv('label_gemini_vi_v2.csv')
# df.dropna(inplace=True)
# print(df.shape)
# df.reset_index(inplace=True)
# df['text'] = df['text'].apply(lambda x: remove_number(x))
# dataloader = CreateDataset(df['text'], df['sentiment'], df['classification'],"vinai/phobert-base-v2",128, shuffle=False).label()
# sentences_index = []
# sentiments = []
# classifications = []

# model.eval()
# with torch.no_grad():
#     for sentence, input_ids, attention_mask in tqdm(dataloader):
#         sentences_index.extend(sentence.detach().cpu().numpy())
#         input_ids = input_ids.cuda()
#         attention_mask = attention_mask.cuda()
        
#         sen, clas = model(input_ids, attention_mask)
#         sen = sen.detach().cpu().numpy()
#         clas = clas.detach().cpu().numpy()
        
#         sen = sen.argmax(axis=1).flatten()
#         clas = clas.argmax(axis=1).flatten()
        
#         sentiments.extend(sen)
#         classifications.extend(clas)
        
# sentences_index = np.array(sentences_index)
# sentences = df['text'][sentences_index]
# sentiments = np.array(sentiments)
# classifications = np.array(classifications)
# new_df = pd.DataFrame({'text':sentences, 'sentiment':sentiments, 'classification':classifications})
# new_df.to_csv('label_gemini_vi_v2_linear3head_v2.csv', index=False)


# import torch
# from trainer.base_trainer import Trainer
# from utils.dataloader import CreateDataset
# import pandas as pd

# test_set  = pd.read_csv('dataset/test_set.csv')
# task = 'sentiment'
# bert_name = 'xlm-roberta-base'
# trainer = Trainer(bert_name, task)
# test_data_loader  = CreateDataset(test_set['text'], test_set['sentiment'],test_set['classification'], bert_name, batch_size=128).todataloader()
# bert_name = bert_name.split('/')[-1]
# acc, f1m, f1w =  trainer.evaluate(test_data_loader, save_name=f"{bert_name}-{task}")
# print(f'Final Prediction\n Model :{bert_name} | {task}\nAcc: {acc}, f1m: {f1m}, f1w: {f1w}')

import torch
# from trainer.mlm_head_trainer import Trainer
from trainer.head_trainer import Trainer
from architecture.bert2head.model import BertLinear2HEAD
from utils.dataloader import CreateDataset
import pandas as pd
import gc

bert_name = 'vinai/phobert-base-v2'
batch_size = 128
if 'vinai' in bert_name:
    train_set = pd.read_csv('dataset/train_set_processed.csv')
    test_set  = pd.read_csv('dataset/test_set_processed.csv')
    val_set = pd.read_csv('dataset/val_set_processed.csv')
else:
    train_set = pd.read_csv('dataset/train_set.csv')
    test_set  = pd.read_csv('dataset/test_set.csv')
    val_set = pd.read_csv('dataset/val_set.csv')
    

dataloader  = CreateDataset(val_set['text'], val_set['sentiment'],val_set['classification'], bert_name, batch_size=batch_size).todataloader()
# trainer=Trainer(bert_name,  train_data_loader, val_data_loader)
# bert_name = bert_name.split('/')[-1]

# valid_loss, valid_accs, valid_f1s = trainer.eval(test_data_loader, f"{bert_name}-epoch17")

# print(f'\tVal.acc se : {valid_accs[0]*100:.2f}% | Val.acc ca : {valid_accs[1]*100:.2f}%')
# print(f'\tVal.F1m se : {valid_f1s[0]*100:.2f}  | Val.F1m ca : {valid_f1s[1]*100:.2f}')
# print(f'\tVal.F1w se : {valid_f1s[2]*100:.2f}  | Val.F1w ca : {valid_f1s[3]*100:.2f}')
model = BertLinear2HEAD(bert_name)
model.load_state_dict(torch.load('models/linear/phobert-base-v2-epoch6.pt'))
model = model.to('cuda')
model.eval()
sentiments = []
classifications = []
sen_true = []
clas_true = []

with torch.no_grad():
    for input_ids, attention_mask, b_sent, b_class in dataloader:

        input_ids = input_ids.cuda()
        attention_mask = attention_mask.cuda()
        
        sen, clas = model(input_ids, attention_mask)
        sen = sen.detach().cpu().numpy()
        clas = clas.detach().cpu().numpy()
        
        sen = sen.argmax(axis=1).flatten()
        clas = clas.argmax(axis=1).flatten()
        b_sent = b_sent.cpu().numpy()
        b_class = b_class.cpu().numpy()
        
        sentiments.extend(sen)
        classifications.extend(clas)
        sen_true.extend(b_sent)
        clas_true.extend(b_class)
        
df = pd.DataFrame({'sentiment':sen_true, 'classification':clas_true, 'sentiment_pred':sentiments, 'classification_pred':classifications})
df.to_csv('phobert-base-v2-epoch6_val.csv', index=False)