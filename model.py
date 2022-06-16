import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import BertTokenizer, BertModel
from transformers import RobertaTokenizer, RobertaModel
from transformers import LukeTokenizer, LukeModel
from allennlp.modules.elmo import Elmo, batch_to_ids

from keras.preprocessing.sequence import pad_sequences


class Model(nn.Module):

    def __init__(self, dataset_loader, params):
        super(Model, self).__init__()

        self.params = params
        self.dataset_loader = dataset_loader

        self.wb_method = params.wb_method.lower()

        if self.wb_method == 'glove':
            self.word2id = np.load(os.path.join(params.glove_dir, 'word2id.npy'), allow_pickle=True).tolist()
            self.embedding = nn.Embedding(params.vocab_size, params.glove_dim)
            emb = torch.from_numpy(np.load(os.path.join(params.glove_dir, 'glove_{}d.npy'.format(params.glove_dim)), allow_pickle=True))

            if params.cuda:
                emb = emb.cuda()
            self.embedding.weight.data.copy_(emb)

            params.embedding_dim = params.glove_dim

        elif self.wb_method == 'elmo':
            self.embedding = Elmo(os.path.join(params.elmo_dir, params.elmo_options_file), 
                            os.path.join(params.elmo_dir, params.elmo_weight_file), 1)

            for param in self.embedding.parameters():
                param.requires_grad = False

            params.embedding_dim = params.elmo_dim

        elif self.wb_method == 'bert':
            self.embedding = BertModel.from_pretrained("bert-large-cased")
            self.tokenizer = BertTokenizer.from_pretrained("bert-large-cased")

            for param in self.embedding.parameters():
                param.requires_grad = False                 

            params.max_sen_len += 2     # "[CLS]", "[SEP]"
            params.embedding_dim = params.bert_dim

        elif self.wb_method == 'roberta':
            self.embedding = RobertaModel.from_pretrained("roberta-large")
            self.tokenizer = RobertaTokenizer.from_pretrained("roberta-large")

            for param in self.embedding.parameters():
                param.requires_grad = False  

            params.embedding_dim = params.roberta_dim

        elif self.wb_method == 'luke':
            self.embedding = LukeModel.from_pretrained("studio-ousia/luke-large")
            self.tokenizer = LukeTokenizer.from_pretrained("studio-ousia/luke-large")

            for param in self.embedding.parameters():
                param.requires_grad = False                 

            params.max_sen_len += 2     # "<s>", "</s>"
            params.embedding_dim = params.luke_dim


        else:
            print("init: wb_method nie zostala wybrana.")
        

        self.nn_method = params.nn_method.lower()
        if self.nn_method == 'lstm':
            self.lstm = nn.LSTM(params.embedding_dim, params.hidden_dim, batch_first=True)
            self.fc = nn.Linear(params.hidden_dim, params.num_of_tags)

        elif self.nn_method == 'rnn':
            self.rnn = nn.RNN(params.embedding_dim, params.hidden_dim, batch_first=True)
            self.fc = nn.Linear(params.hidden_dim, params.num_of_tags)

        else:
            print("init: nn_method nie zostala wybrana. ")

    def forward(self, sentences, labels):
        
        if self.wb_method == "glove":
            max_len = max(map(lambda x: len(x), sentences))
            sentences = list(map(lambda x: list(map(lambda w: self.word2id.get(w, 0), x)), sentences))
            sentences = list(map(lambda x: x + [self.params.vocab_size-1] * (max_len - len(x)), sentences))

            sentences = pad_sequences([[w for w in sen] for sen in sentences],
                          maxlen=self.params.max_sen_len, dtype="long", truncating="post", padding="post")


            labels = pad_sequences([[l for l in lab] for lab in labels],
                maxlen=self.params.max_sen_len, value=self.params.pad_tag_num, padding="post",       #self.tags[self.params.pad_tag]   self.params.pad_tag_num
                dtype="long", truncating="post")


            sentences = torch.LongTensor(sentences)
            if self.params.cuda:
                sentences = sentences.cuda()

            x = self.embedding(sentences)
        
        elif self.wb_method == 'elmo':
            # sentences = pad_sequences([[w for w in sen] for sen in sentences],
            #               maxlen=self.params.max_sen_len, value=self.params.pad_word, dtype="long", truncating="post", padding="post")
            sentences_padded = []
            tmp_sen = []
            for sen in sentences:
                for i in range(self.params.max_sen_len):
                    if i < len(sen):
                        tmp_sen.append(sen[i])
                    else:
                        tmp_sen.append(self.params.pad_word)
                sentences_padded.append(tmp_sen)
                tmp_sen = []

            sentences = batch_to_ids(sentences_padded)

            labels = pad_sequences([[l for l in lab] for lab in labels],
                maxlen=self.params.max_sen_len, value=self.params.pad_tag_num, padding="post",       #self.tags[self.params.pad_tag]   self.params.pad_tag_num
                dtype="long", truncating="post")

            sentences = torch.LongTensor(sentences)
            if self.params.cuda:
                sentences = sentences.cuda()
            x = self.embedding(sentences)['elmo_representations'][0]


        elif self.wb_method == 'bert':
            # tokenized_sentences = []
            # tokenized_sen = []
            # tokenized_word = []
            
            # tokenized_labels = []
            # tokenized_sen_labels = []
            # idx = -1

            # for sen, lab in zip(sentences, labels):
            #     idx = -1
            #     for word in sen:
            #         idx += 1
            #         tokenized_word = self.tokenizer.tokenize(word)

            #         for token in tokenized_word:
            #             tokenized_sen.append(token)
            #             tokenized_sen_labels.append(lab[idx])

            #     tokenized_sentences.append(tokenized_sen)
            #     tokenized_labels.append(tokenized_sen_labels)
            #     tokenized_sen = []
            #     tokenized_sen_labels = []

            tokenized_sentences = []
            tokenized_sen = []
            tokenized_word = []
            
            tokenized_labels = []
            tokenized_sen_labels = []
            idx = -1
            is_first = True

            for sen, lab in zip(sentences, labels):
                if sen[0] != "[CLS]":
                    sen.insert(0, "[CLS]")
                    sen.append("[SEP]")
                    lab.insert(0, -1)
                    lab.append(-1)
                idx = -1
                for word in sen:
                    idx += 1
                    tokenized_word = self.tokenizer.tokenize(word)

                    is_first = True
                    for token in tokenized_word:
                        tokenized_sen.append(token)
                        if is_first:
                            tokenized_sen_labels.append(lab[idx])
                            is_first = False
                        else:
                            tokenized_sen_labels.append(-1)

                tokenized_sentences.append(tokenized_sen)
                tokenized_labels.append(tokenized_sen_labels)
                tokenized_sen = []
                tokenized_sen_labels = []

            labels = tokenized_labels
            labels = pad_sequences([[l for l in lab] for lab in labels],
                maxlen=self.params.max_sen_len, value=self.params.pad_tag_num, padding="post",       #self.tags[self.params.pad_tag]   self.params.pad_tag_num
                dtype="long", truncating="post")

            mask = (labels >= 0)
            mask = torch.FloatTensor(mask)
            if self.params.cuda:
                mask = mask.cuda()

            inputs = pad_sequences([self.tokenizer.convert_tokens_to_ids(sen) for sen in tokenized_sentences],
                          maxlen=self.params.max_sen_len, dtype="long", truncating="post", padding="post")

            inputs = torch.LongTensor(inputs)
            if self.params.cuda:
                inputs = inputs.cuda()

            x = self.embedding(inputs, attention_mask = mask)[0]

        elif self.wb_method == 'roberta':

            tokenized_sentences = []
            tokenized_sen = []
            tokenized_word = []
            
            tokenized_labels = []
            tokenized_sen_labels = []
            idx = -1
            is_first = True

            for sen, lab in zip(sentences, labels):
                if sen[0] != "<s>":            
                    sen.insert(0, "<s>")
                    sen.append("</s>")
                    lab.insert(0, -1)
                    lab.append(-1)
                idx = -1
                for word in sen:
                    idx += 1
                    tokenized_word = self.tokenizer.tokenize(word)

                    is_first = True
                    for token in tokenized_word:
                        tokenized_sen.append(token)
                        if is_first:
                            tokenized_sen_labels.append(lab[idx])
                            is_first = False
                        else:
                            tokenized_sen_labels.append(-1)

                tokenized_sentences.append(tokenized_sen)
                tokenized_labels.append(tokenized_sen_labels)
                tokenized_sen = []
                tokenized_sen_labels = []

            labels = tokenized_labels
            labels = pad_sequences([[l for l in lab] for lab in labels],
                maxlen=self.params.max_sen_len, value=self.params.pad_tag_num, padding="post",       #self.tags[self.params.pad_tag]   self.params.pad_tag_num
                dtype="long", truncating="post")

            mask = (labels >= 0)
            mask = torch.FloatTensor(mask)
            if self.params.cuda:
                mask = mask.cuda()

            inputs = pad_sequences([self.tokenizer.convert_tokens_to_ids(sen) for sen in tokenized_sentences],
                          maxlen=self.params.max_sen_len, dtype="long", truncating="post", padding="post")

            inputs = torch.LongTensor(inputs)
            if self.params.cuda:
                inputs = inputs.cuda()

            x = self.embedding(inputs)[0]

        elif self.wb_method == 'luke':
            tokenized_sentences = []
            tokenized_sen = []
            tokenized_word = []
            
            tokenized_labels = []
            tokenized_sen_labels = []
            idx = -1

            for sen, lab in zip(sentences, labels):
                idx = -1
                for word in sen:
                    idx += 1
                    tokenized_word = self.tokenizer.tokenize(word)

                    for token in tokenized_word:
                        tokenized_sen.append(token)
                        tokenized_sen_labels.append(lab[idx])

                tokenized_sentences.append(tokenized_sen)
                tokenized_labels.append(tokenized_sen_labels)
                tokenized_sen = []
                tokenized_sen_labels = []

            labels = tokenized_labels
            labels = pad_sequences([[l for l in lab] for lab in labels],
                maxlen=self.params.max_sen_len, value=self.params.pad_tag_num, padding="post",       #self.tags[self.params.pad_tag]   self.params.pad_tag_num
                dtype="long", truncating="post")

            mask = (labels >= 0)
            mask = torch.FloatTensor(mask)
            if self.params.cuda:
                mask = mask.cuda()

            inputs = pad_sequences([self.tokenizer.convert_tokens_to_ids(sen) for sen in tokenized_sentences],
                          maxlen=self.params.max_sen_len, dtype="long", truncating="post", padding="post")

            inputs = torch.LongTensor(inputs)
            if self.params.cuda:
                inputs = inputs.cuda()

            x = self.embedding(inputs)[0]

        else:
            print("forward: wb_method nie zostala wybrana. ")


        if self.nn_method == 'lstm':
            x, _ = self.lstm(x)
            x = x.contiguous()
            x = x.view(-1, x.shape[2])
            x = self.fc(x)

        elif self.nn_method == 'rnn':
            x, _ = self.rnn(x)
            x = x.contiguous()
            x = x.view(-1, x.shape[2])
            x = self.fc(x)

        else:
            print("forward: nn_method nie zostala wybrana")

        return F.log_softmax(x, dim=1), labels
