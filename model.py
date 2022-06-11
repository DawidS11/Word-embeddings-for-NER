import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import BertTokenizer, BertForTokenClassification

from keras.preprocessing.sequence import pad_sequences


class Model(nn.Module):

    def __init__(self, data_loader, params):
        super(Model, self).__init__()

        self.params = params
        self.data_loader = data_loader

        self.wb_method = params.wb_method.lower()
        if params.wb_method == 'bert':
            self.embedding = BertForTokenClassification.from_pretrained("bert-base-cased", num_labels=params.num_of_tags)
            self.tokenizer = BertTokenizer.from_pretrained("bert-base-cased", num_labels=params.num_of_tags)                  

        elif params.wb_method == 'glove':
            pass

        else:
            print("init: wb_method nie zostala wybrana.")
        

        self.nn_method = params.nn_method.lower()
        if self.nn_method == 'bi-lstm':
            self.lstm = nn.LSTM(params.embedding_dim, params.lstm_hidden_dim, batch_first=True)
            self.fc = nn.Linear(params.lstm_hidden_dim, params.num_of_tags)

        else:
            print("init: nn_method nie zostala wybrana. ")

    def forward(self, sentences, labels):
        
        if self.wb_method == 'bert':
            
            # Removing PAD elements:
            # for i in range(len(labels)):
            #     while(labels[i][-1] == self.params.pad_tag_num):
            #         labels[i] = labels[i][:-1]

            sentences_lists = [' '.join(x) for x in sentences]
            tokenized_text = [self.tokenizer.tokenize(sen) for sen in sentences_lists]

#             tokenized_text = []
#             tokenized_word = []
#             tokenized_sen = []
#             lab_len = 1
#             i = -1
#             j = -1

#             for sen in sentence_lists:
#                 #print(" ")
#                 i += 1
#                 j = -1
#                 tmp_len = len(sen)
#                 for w in sen.split(' '):
#                     j += 1
#                     tokenized_word = self.tokenizer.tokenize(w)
#                     tmp_len += len(tokenized_word)
#                     tmp_len -= 1
#                     for word in tokenized_word:
#                         tokenized_sen.append(word)

#                     while(lab_len < len(tokenized_word)):
#                         if (len(labels[i]) + len(tokenized_word) - 1) == tmp_len:
#                             labels[i] = np.append(labels[i], labels[i][-1])
#                         else:
#                             # print(labels[i])
#                             # print(len(labels[i]))
#                             # print(sen)
#                             # print(len(sen.split(' ')))
#                             # print(tokenized_sen)
#                             # print(len(tokenized_sen))
#                             # print(tokenized_word)
#                             labels[i] = np.insert(labels[i], j+1, labels[i][j])
#                         lab_len += 1
#                     lab_len = 1
#                 tokenized_text.append(tokenized_sen)
#                 tokenized_sen = []

        
            # labels = pad_sequences([[l for l in lab] for lab in labels],
            #           maxlen=self.params.max_sen_len, value=self.tags[self.params.pad_tag], padding="post",       #self.tags[self.params.pad_tag]   self.params.pad_tag_num
            #           dtype="long", truncating="post")

            inputs = pad_sequences([self.tokenizer.convert_tokens_to_ids(sen) for sen in tokenized_text],
                          maxlen=self.params.max_sen_len, dtype="long", truncating="post", padding="post")

            inputs = torch.LongTensor(inputs)
            if self.params.cuda:
                inputs = inputs.cuda()

            x = self.embedding(inputs)[0]
            
        elif self.wb_method == "glove":
            pass   

        else:
            print("forward: wb_method nie zostala wybrana. ")

        if self.nn_method == 'bi-lstm':
            x, _ = self.lstm(x)
            x, _ = self.lstm(x)
            x = x.contiguous()
            x = x.view(-1, x.shape[2])
            x = self.fc(x)

        else:
            print("forward: nn_method nie zostala wybrana")

        return F.log_softmax(x, dim=1), labels