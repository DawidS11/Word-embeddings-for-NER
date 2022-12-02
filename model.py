import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from allennlp.modules.elmo import Elmo, batch_to_ids
from transformers import BertTokenizer, BertModel
from transformers import RobertaTokenizer, RobertaModel
from transformers import LukeTokenizer, LukeModel
from keras_preprocessing.sequence import pad_sequences

from prepare_labels import calc_entity_spans
from dataset_loader import prepare_elmo, prepare_bert_roberta

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

            #if params.cuda:
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

            max_len = max(map(lambda x: len(x), sentences))#, default=0)                                                                          # DEFAULT
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
            sentences = [contexts[idx]['context_text'] for idx in range(len(contexts))]         # tu nie bedzie sentence?
            #labels = [contexts[idx]['context_labels'] for idx in range(len(contexts))]
            
            all_entities = calc_entity_spans(contexts)

            entities = []
            entity_spans = []
            entity_labels = []
            for context in all_entities:
                context_entities = []
                context_spans = []
                context_labels = []
                for entity in context:
                    context_entities.append(entity['entity_text'])
                    context_spans.append(entity['entity_span'])
                    context_labels.append(self.val2id_entity[entity['entity_label']])
                entities.append(context_entities)
                entity_spans.append(context_spans)
                entity_labels.append(context_labels)

            # max_num = max([len(l) for l in entity_labels])
            # entity_labels = pad_sequences([[l for l in lab] for lab in entity_labels],
            #     maxlen=max_num, value=self.params.pad_tag_num, padding="post",
            #     dtype="long", truncating="post")

            texts = [" ".join(sen) for sen in sentences]
            
            inputs_emb = self.tokenizer(texts, entities=entities, entity_spans=entity_spans, return_tensors="pt", padding=True)

            # inputs = inputs_emb
            # inputs2 = inputs["input_ids"].long()
            # attention_mask2 = inputs["attention_mask"].long()
            # entity_attention_mask = inputs["entity_attention_mask"].long()
            # entity_ids = inputs["entity_ids"].long()
            # entity_position_ids = inputs["entity_position_ids"].long()

            # inputs = torch.LongTensor(inputs2)
            # attention_mask = torch.LongTensor(attention_mask2)
            # entity_attention_mask = torch.LongTensor(entity_attention_mask)8
            # entity_position_ids = torch.LongTensor(entity_position_ids)

            # if self.params.cuda:
            #     inputs = inputs.cuda()
            #     attention_mask = attention_mask.cuda() 
            #     entity_attention_mask = entity_attention_mask.cuda()
            #     entity_ids = entity_ids.cuda()
            #     entity_position_ids = entity_position_ids.cuda()

            # outputs = self.embedding(inputs, attention_mask=attention_mask, entity_attention_mask=entity_attent ion_mask, entity_ids=entity_ids, entity_position_ids=entity_position_ids)
            inputs_emb = inputs_emb.to(device=self.params.device)
            outputs = self.embedding(**inputs_emb)
            del inputs_emb
            x = outputs['entity_last_hidden_state']
            labels = entity_labels

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

        return F.log_softmax(x, dim=1), labels
