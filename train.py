import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
from tqdm import trange
import time

from model import Model
from dataset_loader import DatasetLoader
from keras.preprocessing.sequence import pad_sequences

import params

def loss_fun_luke(outputs, labels, id2val_entity):
    pass


def accuracy_luke(outputs, labels, id2val_entity, val2id):

    labels = labels.ravel()         # (1D: batch_size*seq_len)

    mask = (labels >= 0)

    outputs = np.argmax(outputs, axis=1)
    outputs = [id2val_entity[l] if l < 5 else "WRONG" for l in outputs]

    
    if float(np.sum(mask)) > 0.0:
        acc = np.sum(outputs == labels)/float(np.sum(mask)) 
    else:
        acc = 0
    return acc


def loss_fun(outputs, labels):
    
    labels = labels.ravel()         # (1D: batch_size*seq_len)

    mask = (labels >= 0).float()

    labels = labels % outputs.shape[1]

    num_tokens = int(torch.sum(mask))

    return -torch.sum(outputs[range(outputs.shape[0]), labels]*mask)/num_tokens


def accuracy(outputs, labels):

    labels = labels.ravel()         # (1D: batch_size*seq_len)

    mask = (labels >= 0)

    outputs = np.argmax(outputs, axis=1)

    if float(np.sum(mask)) > 0.0:
        acc = np.sum(outputs == labels)/float(np.sum(mask)) 
    else:
        acc = 0
    return acc


def evaluate(model, criterion, criterion_luke, data_eval_iterator, num_batches, params, id2val_entity):
     
    model.eval()
    total_loss = 0.0
    total_acc = 0.0
    with torch.no_grad():
        batches = trange(num_batches)
        for batch in batches:

            sentences, labels, contexts = next(data_eval_iterator)

            outputs, labels = model(sentences, labels, contexts)

            labels = torch.LongTensor(labels)
            if params.cuda:
                labels = labels.cuda()

            loss = criterion(outputs, labels)

            outputs = outputs.data.cpu().numpy()
            labels = labels.data.cpu().numpy()

            total_loss += loss.item()
            total_acc += accuracy(outputs, labels)

    avg_loss = total_loss / num_batches
    avg_acc = total_acc / num_batches
    return avg_loss, avg_acc


def train(model, optimizer, criterion, criterion_luke, data_train_iterator, num_batches, params, id2val_entity, val2id):

    model.train()
    total_loss = 0.0
    total_acc = 0.0

    batches = trange(num_batches)
    for batch in batches:

        sentences, labels, contexts = next(data_train_iterator)

        # print(num_ent)
        # if params.we_method.lower() == 'luke':
        #     original_labels = [contexts[idx]['labels'] for idx in range(len(contexts))]
        #     max_num = max([len(l) for l in original_labels])
        #     # original_labels = pad_sequences([[l for l in lab] for lab in original_labels],
        #     #     maxlen=max_num, value=params.pad_tag_num, padding="post",  
        #     #     dtype="long", truncating="post")
        #     # original_labels = torch.LongTensor(original_labels)
        #     # if params.cuda:
        #     #     original_labels = original_labels.cuda()

        #     predicted_sequences = []
        #     for i in range(len(original_labels)):
        #         predicted_sequence = ["O"] * len(original_labels[i])
        #         predicted_sequences.append(predicted_sequence)

        outputs, labels = model(sentences, labels, contexts)

        if params.we_method.lower() == 'luke':                      # odczytanie labeli encji
            out = outputs.data.cpu().numpy()
            out = np.argmax(out, axis=1)
            out_len = len(out)                  # 6560
            print(out_len)
            original_labels = [contexts[idx]['labels'] for idx in range(len(contexts))]
            num_entities = int(out_len/len(original_labels))            # liczba encji w batch
            max_num = 0                                                 # liczba każdej z pary (a, b) 
            for i in range(num_entities):
                if (i*i - i*(i-1)/2) == num_entities:
                    max_num = i
                    break
            num = []
            for j in range(len(original_labels)):
                for i in range(len(labels[j])):
                    if (i*i - i*(i-1)/2) == len(labels[j]):
                        num.append(i)
                        break
            original_labels = pad_sequences([[l for l in lab] for lab in original_labels],
                maxlen=max_num, value=params.pad_tag_num, padding="post",  
                dtype="long", truncating="post")

            predicted_sequences = []
            for i in range(len(original_labels)):
                predicted_sequence = ["O"] * len(original_labels[i])
                predicted_sequences.append(predicted_sequence)

            idx = 0
            for i in range(len(original_labels)):
                for j in range(num[i]):
                    for k in range(j, num[i]):
                        entity_val = id2val_entity[out[idx]] if out[idx] < 5 else "WRONG"
                        if entity_val == "NIL":
                            idx += 1
                            continue
                        else:
                            idx += 1
                            predicted_sequences[i][j] = "B-" + entity_val
                            if k > j:
                                for l in range(j+1, k+1):
                                    predicted_sequences[i][l] = "I-" + entity_val
                for j in range(num[i], max_num):
                    for k in range(j, max_num):
                        idx += 1
                                
                        
            print(idx)      # ma byc 6503
            print(predicted_sequences)
            quit()

        else:
            labels = torch.LongTensor(labels)
            if params.cuda:
                labels = labels.cuda()

        if params.we_method.lower() == 'a':
            loss = criterion_luke(outputs, original_labels, id2val_entity)
        else:
            loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()

        optimizer.step()

        outputs = outputs.data.cpu().numpy()
        labels = labels.data.cpu().numpy()

        total_loss += loss.item()
        total_acc += accuracy_luke(outputs, labels, id2val_entity, val2id)

        batches.set_postfix(accuracy='{:05.3f}'.format(total_acc/(batch+1)), loss='{:05.3f}'.format(total_loss/(batch+1)))

    avg_loss = total_loss / num_batches
    avg_acc = total_acc / num_batches

    return avg_loss, avg_acc



if __name__ == '__main__':

    params = params.Params()

    params.cuda = torch.cuda.is_available()
    torch.manual_seed(params.seed)
    if params.cuda:
        torch.cuda.manual_seed(params.seed)

    # Getting data:
    dataset_loader = DatasetLoader(params)
    id2val = dataset_loader.id2val
    val2id = dataset_loader.val2id
    id2val_entity = dataset_loader.id2val_entity
    val2id_entity = dataset_loader.val2id_entity
    data_train = dataset_loader.load_data("train", params)
    data_val = dataset_loader.load_data("val", params)

    model = Model(params, id2val, val2id, val2id_entity).cuda() if params.cuda else Model(params, id2val, val2id, val2id_entity)
    optimizer = optim.Adam(model.parameters(), lr=params.learning_rate)
 
    criterion = loss_fun
    criterion_luke = loss_fun_luke
    

    print("\n\nTraining...")
    best_acc = -1.0
    best_epoch = -1
    best_epoch_loss = -1.0
    total_time = 0.0
    
    for epoch in range(params.num_epochs):
        print("Epoch {}/{}".format(epoch + 1, params.num_epochs), )

        start_train_time = time.time()
        # Training:
        num_batches = (params.train_size + 1) // params.train_batch_size           # number of batches in one epoch
        data_train_iterator = dataset_loader.data_iterator(data_train, params.train_size, params.train_batch_size, params, shuffle=False)
        avg_loss, avg_acc = train(model, optimizer, criterion, criterion_luke, data_train_iterator, num_batches, params, id2val_entity, val2id)

        end_train_time = time.time()
        train_time = end_train_time - start_train_time
        print("Average train loss: ", avg_loss)
        print("Average train accuracy: ", avg_acc)
        print("Training time: ", train_time, "\n")
        total_time += train_time

        # Validation:
        num_batches = (params.val_size + 1) // params.val_batch_size
        data_val_iterator = dataset_loader.data_iterator(data_val, params.val_size, params.val_batch_size, params, shuffle=False)
        avg_loss, avg_acc = evaluate(model, criterion, criterion_luke, data_val_iterator, num_batches, params, id2val_entity)

        end_val_time = time.time()
        val_time = end_val_time - end_train_time
        print("Average val loss: ", avg_loss)
        print("Average val accuracy: ", avg_acc)
        print("Validation time: ", val_time, "\n\n")
        total_time += val_time

        if avg_acc > best_acc:
            best_acc = avg_acc
            best_epoch = epoch
            best_epoch_loss = avg_loss

    print("\nBest accuracy: {:05.3f} for epoch number {} with the loss: {:05.3f}".format(best_acc, best_epoch+1, best_epoch_loss))

    
    print("\n\nTesting...")
    data_test = dataset_loader.load_data("test", params)

    start_test_time = time.time()
    num_batches = (params.test_size + 1) // params.val_batch_size
    data_test_iterator = dataset_loader.data_iterator(data_test, params.test_size, params.val_batch_size, params, shuffle=False)
    avg_loss, avg_acc = evaluate(model, criterion, criterion_luke, data_test_iterator, num_batches, params, id2val_entity)
    end_test_time = time.time()
    test_time = start_test_time - end_test_time
    total_time += test_time

    print("Test accuracy: {:05.3f} with the loss: {:05.3f}".format(avg_acc, avg_loss))
    print("Total time: {:05.3f}". format(total_time))