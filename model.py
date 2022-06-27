import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import gc

from transformers import BertTokenizer, BertModel
from transformers import RobertaTokenizer, RobertaModel
from transformers import LukeTokenizer, LukeModel
from allennlp.modules.elmo import Elmo, batch_to_ids

from keras.preprocessing.sequence import pad_sequences

class Model(nn.Module):

    def __init__(self, params, id2val):
        super(Model, self).__init__()

        self.params = params
        self.id2val = id2val
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
            max_len = max(map(lambda x: len(x), sentences), default=0)                                                                          # DEFAULT
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
        
        elif self.we_method == 'elmo':
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


        elif self.we_method == 'bert':

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
            # labels = pad_sequences([[l for l in lab] for lab in labels],
            #     maxlen=self.params.max_sen_len, value=self.params.pad_tag_num, padding="post",  
            #     dtype="long", truncating="post")
            labels = pad_sequences([[l for l in lab] for lab in labels],
                maxlen=(self.params.max_context_len+2), value=self.params.pad_tag_num, padding="post",  
                dtype="long", truncating="post")
            
            
            mask = (labels >= 0)
            mask = torch.FloatTensor(mask)
            if self.params.cuda:
                mask = mask.cuda()

            # inputs = pad_sequences([self.tokenizer.convert_tokens_to_ids(sen) for sen in tokenized_sentences],
            #                 maxlen=self.params.max_sen_len, dtype="long", truncating="post", padding="post")
            inputs = pad_sequences([self.tokenizer.convert_tokens_to_ids(sen) for sen in tokenized_sentences],
                            maxlen=(self.params.max_context_len+2), dtype="long", truncating="post", padding="post")

            inputs = torch.LongTensor(inputs)
            if self.params.cuda:
                inputs = inputs.cuda()
            
            x = self.embedding(inputs, attention_mask = mask)[0]
            

        elif self.we_method == 'roberta':

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


        elif self.we_method == 'luke':
            original_labels = [context["labels"] for context in contexts]
            original_labels = pad_sequences([[l for l in lab] for lab in labels],
                maxlen=self.params.max_sen_len, value=self.params.pad_tag_num, padding="post",
                dtype="long", truncating="post")

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
                        if is_first:
                            tokenized_sen_labels.append(lab[idx])
                            is_first = False
                        else:
                            tokenized_sen_labels.append(-1)

                tokenized_labels.append(tokenized_sen_labels)
                tokenized_sen_labels = []
            for lab in tokenized_labels:
                if lab[0] != -1:            
                    lab.insert(0, -1)
                    lab.append(-1)

            # entity_spans, empty = calc_entity_spans(sentences, labels, self.id2val)
            entity_spans, word_spans = calc_entity_spans(contexts, self.id2val)
            labels = tokenized_labels
            # labels = pad_sequences([[l for l in lab] for lab in labels],
            #     maxlen=self.params.max_sen_len, value=self.params.pad_tag_num, padding="post",
            #     dtype="long", truncating="post")
            labels = pad_sequences([[l for l in lab] for lab in labels],
                maxlen=(self.params.max_context_len+2), value=self.params.pad_tag_num, padding="post",
                dtype="long", truncating="post")

            texts = [" ".join(sen) for sen in sentences]

            # w przypadku braku encji w batch:
            empty = False
            if empty:
                inputs = self.tokenizer(texts, return_tensors="pt", padding=True)
            else:
                inputs = self.tokenizer(texts, entity_spans=entity_spans, return_tensors="pt", padding=True)

            inputs_emb = inputs
            inputs2 = inputs["input_ids"].long()
            attention_mask2 = inputs["attention_mask"].long()
            if not empty:
                entity_attention_mask = inputs["entity_attention_mask"].long()
                entity_ids = inputs["entity_ids"].long()
                entity_position_ids = inputs["entity_position_ids"].long()

            # inputs = pad_sequences([sen for sen in inputs2],
            #                     maxlen=self.params.max_sen_len, dtype="long", truncating="post", padding="post")
            # attention_mask = pad_sequences([mask for mask in attention_mask2],
            #                     maxlen=self.params.max_sen_len, dtype="long", value=0, truncating="post", padding="post")
            inputs = pad_sequences([sen for sen in inputs2],
                                maxlen=(self.params.max_context_len+2), dtype="long", truncating="post", padding="post")
            attention_mask = pad_sequences([mask for mask in attention_mask2],
                                maxlen=(self.params.max_context_len+2), dtype="long", value=0, truncating="post", padding="post")

            inputs = torch.LongTensor(inputs)
            attention_mask = torch.LongTensor(attention_mask)
            if not empty:
                entity_attention_mask = torch.LongTensor(entity_attention_mask)
                entity_ids = torch.LongTensor(entity_ids)
                entity_position_ids = torch.LongTensor(entity_position_ids)

            if self.params.cuda:
                inputs = inputs.cuda()
                attention_mask = attention_mask.cuda()
                if not empty:
                    entity_attention_mask = entity_attention_mask.cuda()
                    entity_ids = entity_ids.cuda()
                    entity_position_ids = entity_position_ids.cuda()

            # if empty:
            #     outputs = self.embedding(inputs, attention_mask=attention_mask)
            # else:
            #     outputs = self.embedding(inputs, attention_mask=attention_mask, entity_attention_mask=entity_attention_mask, entity_ids=entity_ids, entity_position_ids=entity_position_ids)
            # x = outputs[0]
            # print(outputs)
            # print("\n===============\n")
            if self.params.cuda:
                inputs_emb = inputs_emb.to("cuda")
            outputs = self.embedding(**inputs_emb)
            x = outputs['entity_last_hidden_state']
            print(x.shape)
            x = outputs['last_hidden_state']
            print(x.shape)

            del inputs
            del attention_mask
            if not empty:
                del entity_attention_mask
                del entity_ids
                del entity_position_ids
            # gc.collect()
            # torch.cuda.empty_cache()

        else:
            print("forward: we_method nie zostala wybrana. ")


        if self.nn_method == 'lstm':
            x, _ = self.lstm(x)

        elif self.nn_method == 'rnn':
            x, _ = self.rnn(x)

        else:
            print("forward: nn_method nie zostala wybrana")
        

        # sentence_tensor = x[0:1, contexts[0]["sentence_beg"]:contexts[0]["sentence_end"], :]
        # pad_tensor = torch.zeros(1, (self.params.max_context_len - len(sentence_tensor[0])), self.params.hidden_dim)
        # if self.params.cuda:
        #     pad_tensor = pad_tensor.cuda()
        # sentences_tensor = torch.cat((sentence_tensor, pad_tensor), 1)

        # for i in range(1, (len(x))):
        #     sentence_tensor = x[i:i+1, contexts[i]["sentence_beg"]:contexts[i]["sentence_end"], :]
        #     pad_tensor = torch.zeros(1, (self.params.max_context_len - len(sentence_tensor[0])), self.params.hidden_dim)
        #     if self.params.cuda:
        #         pad_tensor = pad_tensor.cuda()
        #     sentence_tensor = torch.cat((sentence_tensor, pad_tensor), 1)
        #     sentences_tensor = torch.cat((sentences_tensor, sentence_tensor), 0)
        
        sentence_tensor = x[0:1, contexts[0]["sentence_beg"]:contexts[0]["sentence_end"], :]
        pad_tensor = torch.zeros(1, (self.params.max_context_len - len(sentence_tensor[0])), self.params.hidden_dim)
        if self.params.cuda:
            pad_tensor = pad_tensor.cuda()
        sentences_tensor = torch.cat((sentence_tensor, pad_tensor), 1)

        for i in range(1, (len(x))):
            sentence_tensor = x[i:i+1, contexts[i]["sentence_beg"]:contexts[i]["sentence_end"], :]
            pad_tensor = torch.zeros(1, (self.params.max_context_len - len(sentence_tensor[0])), self.params.hidden_dim)
            if self.params.cuda:
                pad_tensor = pad_tensor.cuda()
            sentence_tensor = torch.cat((sentence_tensor, pad_tensor), 1)
            sentences_tensor = torch.cat((sentences_tensor, sentence_tensor), 0)


        x = sentences_tensor
        x = x.contiguous()
        x = x.view(-1, x.shape[2])
        x = self.dropout(x)
        x = self.fc(x)

        return F.log_softmax(x, dim=1), original_labels#, labels



# def calc_entity_spans(sentences, labels, id2val):
#     entity_spans = []
#     sentence_entity_spans = []

#     beg = 0
#     end = 0
#     beg_save = -1
#     is_first_word = True
#     for sen, lab in zip(sentences, labels):

#         # sen = sen["context_text"]
#         # lab = lab["context_labels"]

#         is_first_word = True
#         beg = 0
#         end = 0
#         beg_save = -1
#         for s, l in zip(sen, lab):
#             if is_first_word == False:
#                 end += 1

#             if id2val[l][0] == "B":
#                 beg_save = beg
#                 end += len(s)
#             elif id2val[l][0] == "I":
#                 end += len(s)
#             # if l[0] == "B":
#             #     beg_save = beg
#             #     end += len(s)
#             # elif l[0] == "I":
#             #     end += len(s)
#             else:
#                 # save entity span if beg_save != 0
#                 if beg_save != -1:
#                     sentence_entity_spans.append((beg_save, (end-1)))
#                     beg_save = -1

#                 end += len(s)
#                 beg = end+1

#             is_first_word = False
#         entity_spans.append(sentence_entity_spans)
#         sentence_entity_spans = []
    
#     i = 0
#     empty = False
#     for span in entity_spans:
#         if not span:
#             i += 1
#     if i == len(entity_spans):
#         empty = True
#     return entity_spans, empty


def calc_entity_spans(contexts, id2val):
    possible_entity_spans = []
    context_possible_entity_spans = []
    entity_spans = []
    context_entity_spans = []
    entity_spans_labels = []
    context_entity_spans_labels = []
    entity_spans_words = []
    context_entity_spans_words = []
    word_spans = []
    context_word_spans = []

    all_entities = []
    all_context_entities = []

    for context in contexts:

        beg = 0
        end = 0
        beg_save = -1
        end_save = -1
        text = context['context_text']
        text_labels = context['context_labels']
        tmp = [id2val[lab] for lab in text_labels]
        print(tmp[context['sentence_beg']:context['sentence_end']])
        print(text[context['sentence_beg']:context['sentence_end']])
        for i in range(context['sentence_beg']):            # przesuniecie beg na poczatek zdania w calym tekscie
            beg += len(text[i])
            end += beg
        for idx in range(context['sentence_beg'], context['sentence_end']):
            end = beg
            added = False
            entity = text[idx]
            entity2 = text[idx]
            for idx2 in range(idx, context['sentence_end']):
                end += len(text[idx2])
                if idx != idx2:
                    entity += " "
                    entity += text[idx2]
                if not added:
                    if id2val[text_labels[idx]][0] == 'B':
                        beg_save = beg
                        
                        if id2val[text_labels[idx2]][0] == 'I':
                            end_save = end
                            entity2 += " "
                            entity2 += text[idx2]
                            
                        elif id2val[text_labels[idx2]][0] == 'B':       
                            if idx != idx2:                             # encja o dlugosci jednego slowa poprzedzajaca inna encje (inna niz 'O')
                                if beg_save != -1 and end_save != -1:
                                    context_entity_spans.append((beg_save, end_save))
                                    context_entity_spans_labels.append(id2val[text_labels[idx]][2:])      # zapisz label bez I/B
                                    context_entity_spans_words.append(entity2)
                                    all_context_entities.append(dict(
                                        entity_span=(beg_save, end_save),
                                        entity_label=id2val[text_labels[idx]][2:],
                                        entity_text=entity2
                                    ))
                                    added = True
                                    beg_save = -1
                                    end_save = -1

                        elif id2val[text_labels[idx2]][0] == 'O':
                            if beg_save != -1 and end_save != -1:
                                context_entity_spans.append((beg_save, end_save))
                                context_entity_spans_labels.append(id2val[text_labels[idx]][2:])      # zapisz label bez I/B
                                context_entity_spans_words.append(entity2)
                                all_context_entities.append(dict(
                                    entity_span=(beg_save, end_save),
                                    entity_label=id2val[text_labels[idx]][2:],
                                    entity_text=entity2
                                ))
                                added = True
                                beg_save = -1
                                end_save = -1
                    
                    else:
                        all_context_entities.append(dict(
                            entity_span=(beg, end),
                            entity_label="NIL",
                            entity_text=entity
                        ))
                
                else:
                    all_context_entities.append(dict(
                        entity_span=(beg, end),
                        entity_label="NIL",
                        entity_text=entity
                    ))


                context_possible_entity_spans.append(
                    (beg, end)
                )
                context_word_spans.append(
                    (idx, idx+1)
                )
            beg += len(text[idx])
        # print(context_entity_spans)
        # print(context_entity_spans_labels)
        # print(context_entity_spans_words)
        # quit()

        possible_entity_spans.append(context_possible_entity_spans)
        context_possible_entity_spans = []
        entity_spans.append(context_entity_spans)
        context_entity_spans = []
        entity_spans_labels.append(context_entity_spans_labels)
        context_entity_spans_labels = []
        entity_spans_words.append(context_entity_spans_words)
        context_entity_spans_words = []
        word_spans.append(context_word_spans)
        context_word_spans = []

        all_entities.append(all_context_entities)
        all_context_entities = []

    #print(all_entities)
    #quit()

    return all_entities, word_spans