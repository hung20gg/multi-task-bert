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


import torch
from trainer.base_trainer import Trainer
from utils.dataloader import CreateDataset
import pandas as pd

test_set  = pd.read_csv('dataset/test_set.csv')
task = 'sentiment'
bert_name = 'xlm-roberta-base'
trainer = Trainer(bert_name, task)
test_data_loader  = CreateDataset(test_set['text'], test_set['sentiment'],test_set['classification'], bert_name, batch_size=128).todataloader()
bert_name = bert_name.split('/')[-1]
acc, f1m, f1w =  trainer.evaluate(test_data_loader, save_name=f"{bert_name}-{task}")
print(f'Final Prediction\n Model :{bert_name} | {task}\nAcc: {acc}, f1m: {f1m}, f1w: {f1w}')