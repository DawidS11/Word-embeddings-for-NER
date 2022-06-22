'''
    code: https://colab.research.google.com/github/studio-ousia/luke/blob/master/notebooks/huggingface_conll_2003.ipynb#scrollTo=Y-v01ix-i2FS
'''

import unicodedata

import numpy as np
import seqeval.metrics
import spacy
import torch
import torch.nn as nn
from tqdm import tqdm, trange
from transformers import LukeTokenizer, LukeForEntitySpanClassification, LukeModel

from keras.preprocessing.sequence import pad_sequences
import torch.optim as optim
from train import loss_fun as loss_fun
from train import accuracy as accuracy
import torch.nn.functional as F


def load_documents(dataset_file):
    documents = []
    words = []
    labels = []
    sentence_boundaries = []

    with open(dataset_file) as f:
        for line in f:
            line = line.rstrip()
            if line.startswith("-DOCSTART"):
                if words:
                    documents.append(dict(
                        words=words,
                        labels=labels,
                        sentence_boundaries=sentence_boundaries
                    ))
                    words = []
                    labels = []
                    sentence_boundaries = []
                continue

            if not line:
                if not sentence_boundaries or len(words) != sentence_boundaries[-1]:
                    sentence_boundaries.append(len(words))
            else:
                items = line.split(" ")
                words.append(items[0])
                labels.append(items[-1])

    if words:
        documents.append(dict(
            words=words,
            labels=labels,
            sentence_boundaries=sentence_boundaries
        ))

    return documents


def load_examples(documents, val2id):
    examples = []
    max_token_length = 510
    max_mention_length = 30
    tokenizer = LukeTokenizer.from_pretrained("studio-ousia/luke-base")
    final_labels = []

    for document in tqdm(documents):
        words = document["words"]
        subword_lengths = [len(tokenizer.tokenize(w)) for w in words]

        #tokenizing labels:
        labels = []
        j = -1
        for i in subword_lengths:
            j += 1
            labels.append(val2id[document["labels"][j]])
            for k in range(i-1):
                labels.append(-1)

        final_labels.append(labels)

        total_subword_length = sum(subword_lengths)
        sentence_boundaries = document["sentence_boundaries"]

        for i in range(len(sentence_boundaries) - 1):
            sentence_start, sentence_end = sentence_boundaries[i:i+2]
            if total_subword_length <= max_token_length:
                # if the total sequence length of the document is shorter than the
                # maximum token length, we simply use all words to build the sequence
                context_start = 0
                context_end = len(words)
            else:
                # if the total sequence length is longer than the maximum length, we add
                # the surrounding words of the target sentence　to the sequence until it
                # reaches the maximum length
                context_start = sentence_start
                context_end = sentence_end
                cur_length = sum(subword_lengths[context_start:context_end])
                while True:
                    if context_start > 0:
                        if cur_length + subword_lengths[context_start - 1] <= max_token_length:
                            cur_length += subword_lengths[context_start - 1]
                            context_start -= 1
                        else:
                            break
                    if context_end < len(words):
                        if cur_length + subword_lengths[context_end] <= max_token_length:
                            cur_length += subword_lengths[context_end]
                            context_end += 1
                        else:
                            break

            text = ""
            for word in words[context_start:sentence_start]:
                if word[0] == "'" or (len(word) == 1 and is_punctuation(word)):
                    text = text.rstrip()
                text += word
                text += " "

            sentence_words = words[sentence_start:sentence_end]
            sentence_subword_lengths = subword_lengths[sentence_start:sentence_end]

            word_start_char_positions = []
            word_end_char_positions = []
            for word in sentence_words:
                if word[0] == "'" or (len(word) == 1 and is_punctuation(word)):
                    text = text.rstrip()
                word_start_char_positions.append(len(text))
                text += word
                word_end_char_positions.append(len(text))
                text += " "

            for word in words[sentence_end:context_end]:
                if word[0] == "'" or (len(word) == 1 and is_punctuation(word)):
                    text = text.rstrip()
                text += word
                text += " "
            text = text.rstrip()

            entity_spans = []
            original_word_spans = []
            for word_start in range(len(sentence_words)):
                for word_end in range(word_start, len(sentence_words)):
                    if sum(sentence_subword_lengths[word_start:word_end]) <= max_mention_length:
                        entity_spans.append(
                            (word_start_char_positions[word_start], word_end_char_positions[word_end])
                        )
                        original_word_spans.append(
                            (word_start, word_end + 1)
                        )

            examples.append(dict(
                text=text,
                words=sentence_words,
                entity_spans=entity_spans,
                original_word_spans=original_word_spans,
            ))

    return_labels = pad_sequences([lab for lab in final_labels],
                            maxlen=510, dtype="long", value=-1, truncating="post", padding="post")
    return examples, return_labels


def is_punctuation(char):
    cp = ord(char)
    if (cp >= 33 and cp <= 47) or (cp >= 58 and cp <= 64) or (cp >= 91 and cp <= 96) or (cp >= 123 and cp <= 126):
        return True
    cat = unicodedata.category(char)
    if cat.startswith("P"):
        return True
    return False


class ModelLuke(nn.Module):
    def __init__(self):
        super(ModelLuke, self).__init__()

        self.embedding = LukeModel.from_pretrained("studio-ousia/luke-base")
        self.tokenizer = LukeTokenizer.from_pretrained("studio-ousia/luke-base")

        for param in self.embedding.parameters():
            param.requires_grad = False   

        self.lstm = nn.LSTM(768, 100, batch_first=True)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(100, 9)

    def forward(self, batch_examples):
        # texts = [example["text"] for example in batch_examples]
        texts = [" ".join(example["words"]) for example in batch_examples]

        entity_spans = [example["entity_spans"] for example in batch_examples]

        inputs = self.tokenizer(texts, entity_spans=entity_spans, return_tensors="pt", padding=True)

        attention_mask2 = inputs["attention_mask"]
        entity_attention_mask = inputs["entity_attention_mask"]
        inputs2 = inputs["input_ids"]
        entity_ids = inputs["entity_ids"]
        entity_position_ids = inputs["entity_position_ids"]

        inputs = pad_sequences([sen for sen in inputs2],
                            maxlen=510, dtype="long", truncating="post", padding="post")
        attention_mask = pad_sequences([mask for mask in attention_mask2],
                            maxlen=510, dtype="long", value=0, truncating="post", padding="post")

        if cuda:
            attention_mask = torch.LongTensor(attention_mask)
            entity_attention_mask = torch.LongTensor(entity_attention_mask)
            inputs = torch.LongTensor(inputs)
            entity_ids = torch.LongTensor(entity_ids)
            entity_position_ids = torch.LongTensor(entity_position_ids)
            attention_mask = attention_mask.cuda()
            entity_attention_mask = entity_attention_mask.cuda()
            inputs = inputs.cuda()
            entity_ids = entity_ids.cuda()
            entity_position_ids = entity_position_ids.cuda()

        outputs = self.embedding(inputs, attention_mask = attention_mask, entity_attention_mask=entity_attention_mask, entity_ids=entity_ids, entity_position_ids=entity_position_ids)
        outputs = outputs[0]

        x, _ = self.lstm(outputs)
        x = x.contiguous()
        x = x.view(-1, x.shape[2])
        x = self.dropout(x)
        x = self.fc(x)
        #all_logits.extend(x.logits.tolist())

        return F.log_softmax(x, dim=1)



if __name__ == '__main__':
    cuda = torch.cuda.is_available()
    torch.manual_seed(2022)
    if cuda:
        torch.cuda.manual_seed(2022)

    # Load the model checkpoint
    # model = LukeModel.from_pretrained("studio-ousia/luke-base")
    # model.eval()
    # if cuda:
    #     model.cuda()
    # else:
    #     model()
    # lstm = nn.LSTM(768, 100, batch_first=True)
    # dropout = nn.Dropout(0.3)
    # fc = nn.Linear(100, 9)

    # Load the tokenizer
    # tokenizer = LukeTokenizer.from_pretrained("studio-ousia/luke-base")

    # for param in model.parameters():
    #     param.requires_grad = False  

    batch_size = 2
    all_logits = []
    # train_documents = load_documents("data/conll2003/train.txt")
    # val_documents = load_documents("data/conll2003/valid.txt")
    # test_documents = load_documents("data/conll2003/test.txt")
    train_documents = load_documents("/content/train.txt")
    val_documents = load_documents("/content/valid.txt")
    test_documents = load_documents("/content/test.txt")

    list_labels = []
    for i in range(len(train_documents)):
        for l in train_documents[i]["labels"]:
            list_labels.append(l)
    tags_vals = list(set(list_labels))
    val2id = {t: i for i, t in enumerate(tags_vals)}
    id2val = {i: t for i, t in enumerate(tags_vals)}

    train_examples, train_labels = load_examples(train_documents, val2id)
    val_examples, val_labels = load_examples(val_documents, val2id)
    test_examples, test_labels = load_examples(test_documents, val2id)

    model = ModelLuke().cuda() if cuda else ModelLuke()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = loss_fun


    model.train()
    total_loss = 0.0
    total_acc = 0.0
    num_batches = 0

    for batch_start_idx in trange(0, len(train_examples), batch_size):
        num_batches += 1
        batch_examples = train_examples[batch_start_idx:batch_start_idx + batch_size]
        batch_labels = train_labels[batch_start_idx:batch_start_idx + batch_size]
        outputs = model(batch_examples)         # dlugosc 1020

        batch_labels = torch.LongTensor(batch_labels)
        if cuda:
            batch_labels = batch_labels.cuda()

        loss = criterion(outputs, batch_labels)
        optimizer.zero_grad()
        loss.backward()

        optimizer.step()

        outputs = outputs.data.cpu().numpy()
        batch_labels = batch_labels.data.cpu().numpy()

        total_loss += loss.item()
        total_acc += accuracy(outputs, batch_labels)

    avg_loss = total_loss / num_batches
    avg_acc = total_acc / num_batches
    print("Loss: ", avg_loss, "   Accuracy: ", avg_acc)


    model.eval()
    total_loss = 0.0
    total_acc = 0.0
    num_batches = 0
    for batch_start_idx in trange(0, len(val_examples), batch_size):
        num_batches += 1

        batch_examples = val_examples[batch_start_idx:batch_start_idx + batch_size]
        batch_labels = val_labels[batch_start_idx:batch_start_idx + batch_size]
        outputs = model(batch_examples) 

        batch_labels = torch.LongTensor(batch_labels)
        if cuda:
            batch_labels = batch_labels.cuda()

        loss = criterion(outputs, batch_labels)

        outputs = outputs.data.cpu().numpy()
        batch_labels = batch_labels.data.cpu().numpy()

        total_loss += loss.item()
        total_acc += accuracy(outputs, batch_labels)

    avg_loss = total_loss / num_batches
    avg_acc = total_acc / num_batches
    print("Loss: ", avg_loss, "   Accuracy: ", avg_acc)


    model.eval()
    total_loss = 0.0
    total_acc = 0.0
    num_batches = 0
    for batch_start_idx in trange(0, len(test_examples), batch_size):
        num_batches += 1

        batch_examples = test_examples[batch_start_idx:batch_start_idx + batch_size]
        batch_labels = test_labels[batch_start_idx:batch_start_idx + batch_size]
        outputs = model(batch_examples) 

        batch_labels = torch.LongTensor(batch_labels)
        if cuda:
            batch_labels = batch_labels.cuda()

        loss = criterion(outputs, batch_labels)

        outputs = outputs.data.cpu().numpy()
        batch_labels = batch_labels.data.cpu().numpy()

        total_loss += loss.item()
        total_acc += accuracy(outputs, batch_labels)

    avg_loss = total_loss / num_batches
    avg_acc = total_acc / num_batches
    print("Loss: ", avg_loss, "   Accuracy: ", avg_acc)

    quit()
        # texts = [example["text"] for example in batch_examples]
        # entity_spans = [example["entity_spans"] for example in batch_examples]

        # inputs = tokenizer(texts, entity_spans=entity_spans, return_tensors="pt", padding=True)

        # attention_mask2 = inputs["attention_mask"]
        # entity_attention_mask = inputs["entity_attention_mask"]
        # inputs2 = inputs["input_ids"]
        # entity_ids = inputs["entity_ids"]
        # entity_position_ids = inputs["entity_position_ids"]

        # inputs = pad_sequences([sen for sen in inputs2],
        #                     maxlen=510, dtype="long", truncating="post", padding="post")
        # attention_mask = pad_sequences([mask for mask in attention_mask2],
        #                     maxlen=510, dtype="long", value=0, truncating="post", padding="post")

        # if cuda:
        #     attention_mask = torch.LongTensor(attention_mask)
        #     entity_attention_mask = torch.LongTensor(entity_attention_mask)
        #     inputs = torch.LongTensor(inputs)
        #     entity_ids = torch.LongTensor(entity_ids)
        #     entity_position_ids = torch.LongTensor(entity_position_ids)
        #     attention_mask = attention_mask.cuda()
        #     entity_attention_mask = entity_attention_mask.cuda()
        #     inputs = inputs.cuda()
        #     entity_ids = entity_ids.cuda()
        #     entity_position_ids = entity_position_ids.cuda()

        # outputs = model(inputs, attention_mask = attention_mask, entity_attention_mask=entity_attention_mask, entity_ids=entity_ids, entity_position_ids=entity_position_ids).cuda()
        # outputs = outputs[0]

        # x, _ = lstm(outputs)
        # x = x.contiguous()
        # x = x.view(-1, x.shape[2])
        # x = dropout(x)
        # x = fc(x)
        # all_logits.extend(x.logits.tolist())


    # final_labels = [label for document in test_documents for label in document["labels"]]         # lista labeli calego dokumentu o dlugosci 203621

    final_predictions = []
    for example_index, example in enumerate(test_examples):
        logits = all_logits[example_index]
        max_logits = np.max(logits, axis=1)
        max_indices = np.argmax(logits, axis=1)
        original_spans = example["original_word_spans"]
        predictions = []
        for logit, index, span in zip(max_logits, max_indices, original_spans):
            if index != 0:  # the span is not NIL
                predictions.append((logit, span, model.config.id2label[index]))
                                                                                                    # predictions[last]: [(5.666662693023682, (3, 5), 'ORG'), (9.379473686218262, (10, 11), 'LOC'), (6.54561710357666, (30, 32), 'MISC'), (4.933742523193359, (39, 40), 'PER')]
        # construct an IOB2 label sequence
        predicted_sequence = ["O"] * len(example["words"])
        for _, span, label in sorted(predictions, key=lambda o: o[0], reverse=True):
            if all([o == "O" for o in predicted_sequence[span[0] : span[1]]]):
                predicted_sequence[span[0]] = "B-" + label
                if span[1] - span[0] > 1:
                    predicted_sequence[span[0] + 1 : span[1]] = ["I-" + label] * (span[1] - span[0] - 1)

        final_predictions += predicted_sequence

    print(seqeval.metrics.classification_report([final_labels], [final_predictions], digits=4)) 