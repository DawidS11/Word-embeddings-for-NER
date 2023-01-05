import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from allennlp.modules.elmo import Elmo
from transformers import BertTokenizer, BertModel
from transformers import RobertaTokenizer, RobertaModel
from transformers import LukeTokenizer, LukeForEntitySpanClassification
from keras_preprocessing.sequence import pad_sequences

from dataset_loader import prepare_glove, prepare_elmo, prepare_bert_roberta, prepare_luke

class Model(nn.Module):

    def __init__(self, params, id2val, val2id, id2val_entity, val2id_entity):
        super(Model, self).__init__()

        self.params = params
        self.id2val = id2val
        self.val2id = val2id
        self.id2val_entity = id2val_entity
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

        elif self.we_method == 'bert_base':
            if params.bert_cased:
                self.embedding = BertModel.from_pretrained("bert-base-cased")
                self.tokenizer = BertTokenizer.from_pretrained("bert-base-cased")
            else:
                self.embedding = BertModel.from_pretrained("bert-base-uncased")
                self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

            for param in self.embedding.parameters():
                param.requires_grad = False                 

            params.embedding_dim = params.bert_base_dim
        
        elif self.we_method == 'bert_large':
            if params.bert_cased:
                self.embedding = BertModel.from_pretrained("bert-large-cased")
                self.tokenizer = BertTokenizer.from_pretrained("bert-large-cased")
            else:
                self.embedding = BertModel.from_pretrained("bert-large-uncased")
                self.tokenizer = BertTokenizer.from_pretrained("bert-large-uncased")

            for param in self.embedding.parameters():
                param.requires_grad = False                 

            params.embedding_dim = params.bert_large_dim

        elif self.we_method == 'roberta':
            self.embedding = RobertaModel.from_pretrained("roberta-base")
            self.tokenizer = RobertaTokenizer.from_pretrained("roberta-base")

            for param in self.embedding.parameters():
                param.requires_grad = False  

            params.embedding_dim = params.roberta_dim

        elif self.we_method == 'luke':
            self.embedding = LukeForEntitySpanClassification.from_pretrained("studio-ousia/luke-base")
            self.tokenizer = LukeTokenizer.from_pretrained("studio-ousia/luke-base")

            for param in self.embedding.parameters():
                param.requires_grad = False                 

            params.embedding_dim = params.luke_dim

        elif self.we_method == 'luke_conll':
            self.embedding = LukeForEntitySpanClassification.from_pretrained("studio-ousia/luke-large-finetuned-conll-2003")
            self.tokenizer = LukeTokenizer.from_pretrained("studio-ousia/luke-large-finetuned-conll-2003")

            for param in self.embedding.parameters():
                param.requires_grad = False                 

            params.embedding_dim = params.luke_dim


        else:
            print("init: we_method nie zostala wybrana.")
        
        if params.is_finetuned:
            params.embedding_dim = 1

        self.nn_method = params.nn_method.lower()
        if self.nn_method == 'lstm':
            self.lstm = nn.LSTM(params.embedding_dim, params.hidden_dim, batch_first=True)

        elif self.nn_method == 'rnn':
            self.rnn = nn.RNN(params.embedding_dim, params.hidden_dim, batch_first=True)

        elif self.nn_method == 'cnn':
            self.conv = nn.Conv2d(1, params.hidden_dim, kernel_size=(1, params.embedding_dim))
            nn.init.xavier_uniform_(self.conv.weight)
            nn.init.constant_(self.conv.bias, 0.0)

        else:
            print("init: nn_method nie zostala wybrana. ")

        if self.we_method == 'luke':
            self.fc = nn.Linear(params.hidden_dim, params.num_of_tags_entity)
        else:
            self.fc = nn.Linear(params.hidden_dim, params.num_of_tags)
        nn.init.uniform_(self.fc.weight, -0.5, 0.5)
        nn.init.uniform_(self.fc.bias, -0.1, 0.1)


    def forward(self, sentences, labels, contexts):
        
        if self.we_method == "glove":

            sentences, labels, sentence_begs, sentence_ends = prepare_glove(self.params, self.word2id, contexts)

            inputs = torch.LongTensor(sentences)
            inputs = inputs.to(device=self.params.device)

            x = self.embedding(inputs)
            del inputs

            # Bierze pod uwagę tylko zdanie, nie caly kontekst:
            tmp_x = []
            sentence_labels = []
            for i in range(len(sentence_begs)):
                tmp_x.append(x[i][sentence_begs[i]:sentence_ends[i]])
                sentence_labels.append((labels[i][sentence_begs[i]:sentence_ends[i]]))         
            
            max_num = max([len(a) for a in tmp_x])
            tmp_x = [F.pad(tensor, pad=(0, 0, 0, max_num - tensor.shape[0])) for tensor in tmp_x]
            x = torch.stack(tmp_x)

            max_num = max([len(l) for l in sentence_labels])
            labels = pad_sequences([[l for l in lab] for lab in sentence_labels],
                maxlen=max_num, value=self.params.pad_tag_num, padding="post",       
                dtype="long", truncating="post")
            labels = np.array(labels)
            

        elif self.we_method == 'elmo':

            sentences, labels, sentence_begs, sentence_ends = prepare_elmo(self.params, contexts)

            inputs = torch.LongTensor(sentences)
            inputs = inputs.to(device=self.params.device)

            x = self.embedding(inputs)['elmo_representations'][0]
            del inputs

            # Bierze pod uwagę tylko zdanie, nie caly kontekst:
            tmp_x = []
            sentence_labels = []
            for i in range(len(sentence_begs)):
                tmp_x.append(x[i][sentence_begs[i]:sentence_ends[i]])
                sentence_labels.append((labels[i][sentence_begs[i]:sentence_ends[i]]))         
            
            print(sentence_begs[i], "  ", sentence_ends[i])
            max_num = max([len(a) for a in tmp_x])
            tmp_x = [F.pad(tensor, pad=(0, 0, 0, max_num - tensor.shape[0])) for tensor in tmp_x]
            x = torch.stack(tmp_x)

            max_num = max([len(l) for l in sentence_labels])
            labels = pad_sequences([[l for l in lab] for lab in sentence_labels],
                maxlen=max_num, value=self.params.pad_tag_num, padding="post",       
                dtype="long", truncating="post")
            labels = np.array(labels)

        elif self.we_method == 'bert_base' or self.we_method == 'bert_large':
            
            sentences, labels, sentence_begs, sentence_ends = prepare_bert_roberta(self.params, self.tokenizer, contexts)

            attention_mask = (labels >= 0)
            attention_mask = torch.FloatTensor(attention_mask)
            attention_mask = attention_mask.to(device=self.params.device)

            inputs = torch.LongTensor(sentences)
            inputs = inputs.to(device=self.params.device)

            x = self.embedding(inputs, attention_mask=attention_mask)[0]
            del inputs

            # Bierze pod uwagę tylko zdanie, nie caly kontekst:
            tmp_x = []
            sentence_labels = []
            for i in range(len(sentence_begs)):
                tmp_x.append(x[i][sentence_begs[i]:sentence_ends[i]])
                sentence_labels.append((labels[i][sentence_begs[i]:sentence_ends[i]]))         
            
            max_num = max([len(l) for l in sentence_labels])
            tmp_x = [F.pad(tensor, pad=(0, 0, 0, max_num - tensor.shape[0])) for tensor in tmp_x]
            x = torch.stack(tmp_x)

            labels = pad_sequences([[l for l in lab] for lab in sentence_labels],
                maxlen=max_num, value=self.params.pad_tag_num, padding="post",       
                dtype="long", truncating="post")
            labels = np.array(labels)


        elif self.we_method == 'roberta':

            sentences, labels, sentence_begs, sentence_ends = prepare_bert_roberta(self.params, self.tokenizer, contexts)
            
            attention_mask = (labels >= 0)
            attention_mask = torch.FloatTensor(attention_mask)
            attention_mask = attention_mask.to(device=self.params.device)

            inputs = torch.LongTensor(sentences)
            inputs = inputs.to(device=self.params.device)

            x = self.embedding(inputs, attention_mask=attention_mask)[0]
            del inputs

            # Bierze pod uwagę tylko zdanie, nie caly kontekst:
            tmp_x = []
            sentence_labels = []
            for i in range(len(sentence_begs)):
                tmp_x.append(x[i][sentence_begs[i]:sentence_ends[i]])
                sentence_labels.append((labels[i][sentence_begs[i]:sentence_ends[i]]))         
            
            max_num = max([len(l) for l in sentence_labels])
            tmp_x = [F.pad(tensor, pad=(0, 0, 0, max_num - tensor.shape[0])) for tensor in tmp_x]
            x = torch.stack(tmp_x)

            labels = pad_sequences([[l for l in lab] for lab in sentence_labels],
                maxlen=max_num, value=self.params.pad_tag_num, padding="post",       
                dtype="long", truncating="post")
            labels = np.array(labels)


        elif self.we_method == 'luke' or self.we_method == 'luke_conll':
            
            # sentences, labels, self.params.word_entity_spans = prepare_luke(self.params, contexts, self.tokenizer, self.id2val, self.val2id_entity)

            # inputs = sentences.to(device=self.params.device)
            # outputs = self.embedding(**inputs)        #['entity_last_hidden_state']
            # del inputs

            texts = [example["text"] for example in contexts]
            entity_spans = [example["entity_spans"] for example in contexts]

            inputs = self.tokenizer(texts, entity_spans=entity_spans, return_tensors="pt", padding=True)
            inputs = inputs.to(device=self.params.device)
            with torch.no_grad():
                outputs = self.embedding(**inputs)

            x = outputs.logits.tolist()
            final_predictions = []
            for example_index, example in enumerate(contexts):
                logits = x[example_index]
                max_logits = np.max(logits, axis=1)
                max_indices = np.argmax(logits, axis=1)
                original_spans = example["original_word_spans"]
                predictions = []
                for logit, index, span in zip(max_logits, max_indices, original_spans):
                    if index != 0:  # the span is not NIL
                        predictions.append((logit, span, self.embedding.config.id2label[index]))

                # construct an IOB2 label sequence
                predicted_sequence = ["O"] * len(example["words"])
                for _, span, label in sorted(predictions, key=lambda o: o[0], reverse=True):
                    if all([o == "O" for o in predicted_sequence[span[0] : span[1]]]):
                        predicted_sequence[span[0]] = "B-" + label
                        if span[1] - span[0] > 1:
                            predicted_sequence[span[0] + 1 : span[1]] = ["I-" + label] * (span[1] - span[0] - 1)

                final_predictions.append([self.val2id[l] for l in predicted_sequence])

            max_num = max([len(l) for l in labels])
            labels = pad_sequences([[l for l in lab] for lab in labels],
                maxlen=max_num, value=self.params.pad_tag_num, padding="post",
                dtype="long", truncating="post")
            x = pad_sequences([[l for l in lab] for lab in final_predictions],
                maxlen=max_num, value=self.params.pad_tag_num, padding="post",
                dtype="long", truncating="post")

            x = torch.FloatTensor(x)
            x = x.to(device=self.params.device)
            x = torch.unsqueeze(x, dim=-1)          # Dodanie wymiaru, o rozmiarze 1, który odpowiada za embedding - w tym przypadku id labela
            
        else:
            print("forward: we_method nie zostala wybrana. ")


        if self.nn_method == 'lstm':
            x, _ = self.lstm(x)

        elif self.nn_method == 'rnn':
            x, _ = self.rnn(x)
        
        elif self.nn_method == 'cnn':
            x = x.unsqueeze(1)
            x = F.relu(self.conv(x).squeeze(3))
            x = x.permute(0, 2, 1)

        else:
            print("forward: nn_method nie zostala wybrana")

        x = x.contiguous()
        x = x.view(-1, x.shape[2])
        x = self.dropout(x)
        x = self.fc(x)
        x_log_softmax = F.log_softmax(x, dim=1)
        del x

        return x_log_softmax, labels
