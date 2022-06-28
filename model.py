import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from allennlp.modules.elmo import Elmo, batch_to_ids
from transformers import BertTokenizer, BertModel
from transformers import RobertaTokenizer, RobertaModel
from transformers import LukeTokenizer, LukeModel

from keras.preprocessing.sequence import pad_sequences

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

            if params.cuda:
                emb = emb.cuda()
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
            if self.params.cuda:
                sentences = sentences.cuda()

            x = self.embedding(sentences)
        
        elif self.we_method == 'elmo':
            sentences = [contexts[idx]['context_text'] for idx in range(len(contexts))]
            labels = [contexts[idx]['context_labels'] for idx in range(len(contexts))]
            max_num = max([len(s) for s in sentences])

            sentences_padded = []
            tmp_sen = []
            for sen in sentences:
                for i in range(max_num):
                    if i < len(sen):
                        tmp_sen.append(sen[i])
                    else:
                        tmp_sen.append(self.params.pad_word)
                sentences_padded.append(tmp_sen)
                tmp_sen = []

            sentences = batch_to_ids(sentences_padded)

            labels = pad_sequences([[l for l in lab] for lab in labels],
                maxlen=max_num, value=self.params.pad_tag_num, padding="post",       #self.tags[self.params.pad_tag]   self.params.pad_tag_num
                dtype="long", truncating="post")

            sentences = torch.LongTensor(sentences)
            if self.params.cuda:
                sentences = sentences.cuda()
            x = self.embedding(sentences)['elmo_representations'][0]


        elif self.we_method == 'bert':
            sentences = [contexts[idx]['context_text'] for idx in range(len(contexts))]
            labels = [contexts[idx]['context_labels'] for idx in range(len(contexts))]
            
            tokenized_sentences = []
            tokenized_sen = []
            tokenized_word = []
            
            tokenized_labels = []
            tokenized_sen_labels = []
            idx = -1
            is_first = True

            for sen, lab in zip(sentences, labels):
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
            for sen, lab in zip(tokenized_sentences, tokenized_labels):
                if sen[0] != "[CLS]":
                    sen.insert(0, "[CLS]")
                    sen.append("[SEP]")
                    lab.insert(0, self.params.pad_tag_num)
                    lab.append(self.params.pad_tag_num)

            labels = tokenized_labels
            max_num = max([len(l) for l in labels])
            labels = pad_sequences([[l for l in lab] for lab in labels],
                maxlen=max_num, value=self.params.pad_tag_num, padding="post",  
                dtype="long", truncating="post")
            
            attention_mask = (labels >= 0)
            attention_mask = torch.FloatTensor(attention_mask)
            if self.params.cuda:
                attention_mask = attention_mask.cuda()

            inputs = pad_sequences([self.tokenizer.convert_tokens_to_ids(sen) for sen in tokenized_sentences],
                            maxlen=max_num, dtype="long", truncating="post", padding="post")

            inputs = torch.LongTensor(inputs)
            if self.params.cuda:
                inputs = inputs.cuda()
            
            x = self.embedding(inputs, attention_mask=attention_mask)[0]
            

        elif self.we_method == 'roberta':
            sentences = [contexts[idx]['context_text'] for idx in range(len(contexts))]
            labels = [contexts[idx]['context_labels'] for idx in range(len(contexts))]

            tokenized_sentences = []
            tokenized_sen = []
            tokenized_word = []
            
            tokenized_labels = []
            tokenized_sen_labels = []
            idx = -1
            is_first = True

            for sen, lab in zip(sentences, labels):
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
            for sen, lab in zip(tokenized_sentences, tokenized_labels):
                if sen[0] != "<s>":
                    sen.insert(0, "<s>")
                    sen.append("</s>")
                    lab.insert(0, self.params.pad_tag_num)
                    lab.append(self.params.pad_tag_num)

            labels = tokenized_labels
            max_num = max([len(l) for l in labels])
            labels = pad_sequences([[l for l in lab] for lab in labels],
                maxlen=max_num, value=self.params.pad_tag_num, padding="post",       
                dtype="long", truncating="post")

            attention_mask = (labels >= 0)
            attention_mask = torch.FloatTensor(attention_mask)
            if self.params.cuda:
                attention_mask = attention_mask.cuda()

            inputs = pad_sequences([self.tokenizer.convert_tokens_to_ids(sen) for sen in tokenized_sentences],
                          maxlen=max_num, dtype="long", truncating="post", padding="post")

            inputs = torch.LongTensor(inputs)
            if self.params.cuda:
                inputs = inputs.cuda()

            x = self.embedding(inputs, attention_mask=attention_mask)[0]


        elif self.we_method == 'luke':
            sentences = [contexts[idx]['context_text'] for idx in range(len(contexts))]
            #labels = [contexts[idx]['context_labels'] for idx in range(len(contexts))]

            all_entities = calc_entity_spans(contexts)
            '''
            all_entities:
            batch_size razy all_context_entities:
                dla kazdej mozliwej encji w all_context_entities:
                dict(
                    entity_span=(beg_save, end_save),
                    entity_label=id2val[text_labels[idx]][2:],
                    entity_text=entity2
                ))
            '''
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

            # print(entities[1])
            # quit()
            # a = 0
            # for i in range(len(entity_spans)):
            #     a += len(entity_spans[i])               
            # print(a)
            # quit()

            inputs_emb = self.tokenizer(texts, entities=entities, entity_spans=entity_spans, return_tensors="pt", padding=True)

            # inputs = inputs_emb
            # inputs2 = inputs["input_ids"].long()
            # attention_mask2 = inputs["attention_mask"].long()
            # entity_attention_mask = inputs["entity_attention_mask"].long()
            # entity_ids = inputs["entity_ids"].long()
            # entity_position_ids = inputs["entity_position_ids"].long()

            # inputs = torch.LongTensor(inputs2)
            # attention_mask = torch.LongTensor(attention_mask2)
            # entity_attention_mask = torch.LongTensor(entity_attention_mask)
            # entity_ids = torch.LongTensor(entity_ids)
            # entity_position_ids = torch.LongTensor(entity_position_ids)

            # if self.params.cuda:
            #     inputs = inputs.cuda()
            #     attention_mask = attention_mask.cuda()
            #     entity_attention_mask = entity_attention_mask.cuda()
            #     entity_ids = entity_ids.cuda()
            #     entity_position_ids = entity_position_ids.cuda()

            # outputs = self.embedding(inputs, attention_mask=attention_mask, entity_attention_mask=entity_attention_mask, entity_ids=entity_ids, entity_position_ids=entity_position_ids)

            if self.params.cuda:
                inputs_emb = inputs_emb.to("cuda")
            outputs = self.embedding(**inputs_emb)
            del inputs_emb
            x = outputs['entity_last_hidden_state']
            print(x)
            quit()

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


def calc_entity_spans(contexts):

    all_entities = []
    all_context_entities = []

    for context in contexts:

        beg = 0
        end = 0
        text = context['context_text']

        for i in range(context['sentence_beg']):            # przesuniecie beg na poczatek zdania w calym tekscie
            beg += len(text[i])
            
        for idx in range(context['sentence_beg'], context['sentence_end']):
            end = beg
            entity = text[idx]

            for idx2 in range(idx, context['sentence_end']):
                end += len(text[idx2])
                if idx != idx2:
                    entity += " "
                    entity += text[idx2]

                all_context_entities.append(dict(
                    entity_span=(beg, end),
                    entity_label="NIL",                 # NIL, bo nie jest wykorzystywane
                    entity_text=entity,
                ))

                
            beg += len(text[idx])

        all_entities.append(all_context_entities)
        all_context_entities = []


    return all_entities