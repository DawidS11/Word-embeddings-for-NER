import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from allennlp.modules.elmo import Elmo
from transformers import BertTokenizer, BertModel
from transformers import RobertaTokenizer, RobertaModel
from transformers import LukeTokenizer, LukeModel
from keras_preprocessing.sequence import pad_sequences

from dataset_loader import prepare_elmo, prepare_bert_roberta, prepare_luke

class Model(nn.Module):

    def __init__(self, params, id2val, val2id, val2id_entity):
        super(Model, self).__init__()

        self.params = params
        self.id2val = id2val
        self.val2id = val2id
        self.val2id_entity = val2id_entity
        self.dropout = nn.Dropout(params.dropout)

        self.we_method = params.we_method.lower()

        if self.we_method == 'glove':
            self.word2id = np.load(os.path.join(params.glove_dir, 'word2id.npy'), allow_pickle=True).tolist()
            self.embedding = nn.Embedding(params.vocab_size, params.glove_dim)
            emb = torch.from_numpy(np.load(os.path.join(params.glove_dir, 'glove_{}d.npy'.format(params.glove_dim)), allow_pickle=True))

            emb = emb.to(device=params.device)
            self.embedding.weight.data.copy_(emb)

            params.embedding_dim = params.glove_dim

        elif self.we_method == 'elmo':
            self.embedding = Elmo(os.path.join(params.elmo_dir, params.elmo_options_file), 
                            os.path.join(params.elmo_dir, params.elmo_weight_file), 1)

            for param in self.embedding.parameters():
                param.requires_grad = False

            params.embedding_dim = params.elmo_dim

        elif self.we_method == 'bert':
            self.embedding = BertModel.from_pretrained("bert-base-cased")
            self.tokenizer = BertTokenizer.from_pretrained("bert-base-cased")

            for param in self.embedding.parameters():
                param.requires_grad = False                 

            params.max_sen_len += 2     # "[CLS]", "[SEP]"
            params.embedding_dim = params.bert_dim

        elif self.we_method == 'roberta':
            self.embedding = RobertaModel.from_pretrained("roberta-base")
            self.tokenizer = RobertaTokenizer.from_pretrained("roberta-base")

            for param in self.embedding.parameters():
                param.requires_grad = False  

            params.embedding_dim = params.roberta_dim

        elif self.we_method == 'luke':
            self.embedding = LukeModel.from_pretrained("studio-ousia/luke-base")
            self.tokenizer = LukeTokenizer.from_pretrained("studio-ousia/luke-base")

            for param in self.embedding.parameters():
                param.requires_grad = False                 

            params.max_sen_len += 2     # "<s>", "</s>"
            params.embedding_dim = params.luke_dim


        else:
            print("init: we_method nie zostala wybrana.")
        

        self.nn_method = params.nn_method.lower()
        if self.nn_method == 'lstm':
            self.lstm = nn.LSTM(params.embedding_dim, params.hidden_dim, batch_first=True)
            self.fc = nn.Linear(params.hidden_dim, params.num_of_tags)

        elif self.nn_method == 'rnn':
            self.rnn = nn.RNN(params.embedding_dim, params.hidden_dim, batch_first=True)
            self.fc = nn.Linear(params.hidden_dim, params.num_of_tags)

        else:
            print("init: nn_method nie zostala wybrana. ")

    def forward(self, sentences, labels, contexts):
        
        if self.we_method == "glove":
            sentences = [contexts[idx]['context_text'] for idx in range(len(contexts))]
            labels = [contexts[idx]['context_labels'] for idx in range(len(contexts))]
            max_num = max([len(s) for s in sentences])

            max_len = max(map(lambda x: len(x), sentences), default=0)                                   
            sentences = list(map(lambda x: list(map(lambda w: self.word2id.get(w, 0), x)), sentences))
            sentences = list(map(lambda x: x + [self.params.vocab_size-1] * (max_len - len(x)), sentences))

            sentences = pad_sequences([[w for w in sen] for sen in sentences],
                          maxlen=max_num, dtype="long", truncating="post", padding="post")

            labels = pad_sequences([[l for l in lab] for lab in labels],
                maxlen=max_num, value=self.params.pad_tag_num, padding="post",      
                dtype="long", truncating="post")

            sentences = torch.LongTensor(sentences)
            sentences = sentences.to(device=self.params.device)

            x = self.embedding(sentences)
        

        elif self.we_method == 'elmo':

            sentences, labels = prepare_elmo(self.params, contexts)

            inputs = torch.LongTensor(sentences)
            inputs = inputs.to(device=self.params.device)

            x = self.embedding(inputs)['elmo_representations'][0]


        elif self.we_method == 'bert':
            
            sentences, labels = prepare_bert_roberta(self.params, self.tokenizer, contexts)

            attention_mask = (labels >= 0)
            attention_mask = torch.FloatTensor(attention_mask)
            attention_mask = attention_mask.to(device=self.params.device)

            inputs = torch.LongTensor(sentences)
            inputs = inputs.to(device=self.params.device)
      
            x = self.embedding(inputs, attention_mask=attention_mask)[0]
            del inputs
            

        elif self.we_method == 'roberta':

            sentences, labels = prepare_bert_roberta(self.params, self.tokenizer, contexts)
            
            attention_mask = (labels >= 0)
            attention_mask = torch.FloatTensor(attention_mask)
            attention_mask = attention_mask.to(device=self.params.device)

            inputs = torch.LongTensor(sentences)
            inputs = inputs.to(device=self.params.device)

            x = self.embedding(inputs, attention_mask=attention_mask)[0]
            del inputs


        elif self.we_method == 'luke':
            
            sentences, labels = prepare_luke(self.params, contexts, self.tokenizer, self.id2val, self.val2id_entity)

            # outputs = self.embedding(inputs, attention_mask=attention_mask, entity_attention_mask=entity_attention_mask, entity_ids=entity_ids, entity_position_ids=entity_position_ids)

            inputs = sentences.to(device=self.params.device)
            x = self.embedding(**inputs)['entity_last_hidden_state']
            del inputs

        else:
            print("forward: we_method nie zostala wybrana. ")


        if self.nn_method == 'lstm':
            x, _ = self.lstm(x)

        elif self.nn_method == 'rnn':
            x, _ = self.rnn(x)

        else:
            print("forward: nn_method nie zostala wybrana")

        x = x.contiguous()
        x = x.view(-1, x.shape[2])
        x = self.dropout(x)
        x = self.fc(x)

        x_log_softmax = F.log_softmax(x, dim=1)
        del x

        return x_log_softmax, labels
