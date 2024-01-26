from transformers import AutoModel
import torch.nn as nn
import torch
import torch.nn.functional as F

# @title Finetune Model

class CNN(nn.Module):
    def __init__(self, embedding_dim, output_dim, n_filters=64, filter_sizes= [1,2,3,5],  dropout=0.2,activation=None):

        super().__init__()
        self.ln1= nn.LayerNorm(embedding_dim)
        self.ln2= nn.LayerNorm(len(filter_sizes)*n_filters)
        self.fc_input = nn.Linear(embedding_dim, embedding_dim)
        self.conv_0 = nn.Conv1d(in_channels = embedding_dim,
                                out_channels = n_filters,
                                kernel_size = filter_sizes[0])
        self.conv_1 = nn.Conv1d(in_channels = embedding_dim,
                                out_channels = n_filters,
                                kernel_size = filter_sizes[1])
        self.conv_2 = nn.Conv1d(in_channels = embedding_dim,
                                out_channels = n_filters,
                                kernel_size = filter_sizes[2])
        self.conv_3 = nn.Conv1d(in_channels = embedding_dim,
                                out_channels = n_filters,
                                kernel_size = filter_sizes[3])
        self.activation = nn.SiLU() if activation is None else activation
        self.fc = nn.Linear(len(filter_sizes)*n_filters,output_dim)
        self.linear = nn.Linear(embedding_dim, output_dim)

        self.dropout = nn.Dropout(dropout)
    def forward(self, encoded):

        embedded = self.activation(self.fc_input(encoded))

        embedded = self.ln1(embedded)
        embedded = embedded.permute(0, 2, 1)

        conved_0 = self.activation(self.conv_0(embedded))
        conved_1 = self.activation(self.conv_1(embedded))
        conved_2 = self.activation(self.conv_2(embedded))
        conved_3 = self.activation(self.conv_3(embedded))
        pooled_0 = F.max_pool1d(conved_0, conved_0.shape[2]).squeeze(2)
        pooled_1 = F.max_pool1d(conved_1, conved_1.shape[2]).squeeze(2)
        pooled_2 = F.max_pool1d(conved_2, conved_2.shape[2]).squeeze(2)
        pooled_3 = F.max_pool1d(conved_3, conved_3.shape[2]).squeeze(2)
        cat = self.dropout(torch.cat((pooled_0, pooled_1, pooled_2, pooled_3), dim = 1).cuda())
        # Should we maxpool?
        cat = self.ln2(cat)
        result =  self.fc(cat)
        return result

class CNN_LSTM(nn.Module):
    def __init__(self, embedding_dim,output_dim,  n_filters=64, filter_sizes= [1,2,3,5],  dropout=0.2,LSTM_UNITS=32,activation=None):

        super().__init__()
        self.ln1= nn.LayerNorm(embedding_dim)
        self.ln2= nn.LayerNorm(len(filter_sizes)*n_filters)
        self.fc_input = nn.Linear(embedding_dim,embedding_dim)
        self.conv_0 = nn.Conv1d(in_channels = embedding_dim,
                                out_channels = n_filters,
                                kernel_size = filter_sizes[0])
        self.conv_1 = nn.Conv1d(in_channels = embedding_dim,
                                out_channels = n_filters,
                                kernel_size = filter_sizes[1])
        self.conv_2 = nn.Conv1d(in_channels = embedding_dim,
                                out_channels = n_filters,
                                kernel_size = filter_sizes[2])
        self.conv_3 = nn.Conv1d(in_channels = embedding_dim,
                                out_channels = n_filters,
                                kernel_size = filter_sizes[3])
        self.activation = nn.SiLU() if activation is None else activation

        self.lstm1 = nn.LSTM(len(filter_sizes) * n_filters, LSTM_UNITS, bidirectional=True, batch_first=True)

        self.fc = nn.Linear(len(filter_sizes) * n_filters, output_dim)
        self.fc2 = nn.Linear(LSTM_UNITS*2, output_dim)
        self.dropout = nn.Dropout(dropout)
    def forward(self, encoded):

        embedded = self.activation(self.fc_input(encoded))
        embedded = self.ln1(embedded)
        embedded = embedded.permute(0, 2, 1)
        conved_0 = self.activation(self.conv_0(embedded))
        conved_1 = self.activation(self.conv_1(embedded))
        conved_2 = self.activation(self.conv_2(embedded))
        conved_3 = self.activation(self.conv_3(embedded))
        pooled_0 = F.max_pool1d(conved_0, conved_0.shape[2]).squeeze(2)
        pooled_1 = F.max_pool1d(conved_1, conved_1.shape[2]).squeeze(2)
        pooled_2 = F.max_pool1d(conved_2, conved_2.shape[2]).squeeze(2)
        pooled_3 = F.max_pool1d(conved_3, conved_3.shape[2]).squeeze(2)
        # Should we maxpool?
        cat = self.dropout(torch.cat((pooled_0, pooled_1, pooled_2, pooled_3), dim = 1).cuda())
        cat = self.ln2(cat)
        self.lstm1.flatten_parameters()

        output, (final_hidden_state, final_cell_state) = self.lstm1(cat)

        result =  self.fc2(output)
        return result



class BertCNN(nn.Module):
    def __init__(self, label_size,name):
        super().__init__()
        self.BertModel = AutoModel.from_pretrained(name)
        # self.phoBertModel.load_state_dict(torch.load(pretrained_path))
        self.cnn= CNN(self.BertModel.dimension,label_size)
    def forward(self,sentences,attention):
       embedded = self.BertModel(sentences,attention_mask=attention)[0]
       cnn_predict = self.cnn(embedded)
       return cnn_predict


class BertCNN_LSTM(nn.Module):
    def __init__(self, label_size,name):
        super().__init__()
        self.BertModel = AutoModel.from_pretrained(name)
        # self.phoBertModel.load_state_dict(torch.load(pretrained_path))
        self.cnn= CNN_LSTM(self.BertModel.dimension,label_size)
    def forward(self,sentences,attention):
       embedded = self.BertModel(sentences,attention_mask=attention)[0]
       cnn_predict = self.cnn(embedded)
       return cnn_predict

