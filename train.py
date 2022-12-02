import os

os.environ['KMP_DUPLICATE_LIB_OK']='True'

import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
from tqdm import trange
import time

from model import Model
from dataset_loader import DatasetLoader
from keras_preprocessing.sequence import pad_sequences
from sklearn.metrics import f1_score
import seqeval.metrics
import params


def accuracy_luke(outputs, labels):
    labels = labels.ravel()         # (1D: batch_size*seq_len)
    mask = (labels >= 0)
    outputs = np.array(outputs)

    return np.sum(outputs == labels)/float(np.sum(mask)) 


def stats(outputs, labels, show_table = False):
    labels = labels.ravel()         # (1D: batch_size*seq_len)
    mask = (labels >= 0)
    outputs = np.argmax(outputs, axis=1)
    
    accuracy = np.sum(outputs == labels)/float(np.sum(mask))

    labels_not_masked = []
    outputs_not_masked = []
    for i in range(len(mask)):
        if mask[i]:
            labels_not_masked.append(labels[i])
            outputs_not_masked.append(outputs[i])

    f1 = f1_score(labels_not_masked, outputs_not_masked, average='micro')
    
    labels_vals = [id2val[label] for label in labels_not_masked]
    outputs_vals = [id2val[output] for output in outputs_not_masked]

    # if show_table:
        # labels_vals = [id2val[label] for label in labels_not_masked]
        # outputs_vals = [id2val[output] for output in outputs_not_masked]
        #print("\n", seqeval.metrics.classification_report([labels_vals], [outputs_vals], digits=4)) 

    return accuracy, f1, labels_vals, outputs_vals


def loss_fun(outputs, labels):
    labels = labels.ravel()         # (1D: batch_size*seq_len)
    mask = (labels >= 0).float()
    
    #labels = labels % outputs.shape[1]     # nie dziala z device="mps"
    labels2 = []
    for l in labels:
        labels2.append(l.item() % outputs.shape[1])
    labels = torch.LongTensor(labels2)
    labels = labels.to(device=params.get_device())
    
    num_tokens = int(torch.sum(mask))
    
    return -torch.sum(outputs[range(outputs.shape[0]), labels]*mask)/num_tokens


def accuracy(outputs, labels):
    labels = labels.ravel()         # (1D: batch_size*seq_len)
    mask = (labels >= 0)
    outputs = np.argmax(outputs, axis=1)
    
    return np.sum(outputs == labels)/float(np.sum(mask)) 


def f1(outputs, labels): 
    labels = labels.ravel()
    mask = (labels >= 0)
    outputs = np.argmax(outputs, axis=1)
    labels2 = []
    outputs2 = []
    for i in range(len(mask)):
        if mask[i]:
            labels2.append(labels[i])
            outputs2.append(outputs[i])

    return f1_score(labels2, outputs2, average='micro')


def evaluate(model, criterion, data_eval_iterator, num_batches, params, id2val_entity, show_table = False):
    
    model.eval()
    total_loss = 0.0
    total_acc = 0.0
    total_f1_score = 0.0
    all_labels = []
    all_outputs = []

    with torch.no_grad():
        batches = trange(num_batches)
        for batch in batches:

            sentences, labels, contexts = next(data_eval_iterator)

            outputs, labels = model(sentences, labels, contexts)

            if params.we_method.lower() == 'luke':                      # odczytanie labeli encji
                entity_outputs = outputs
                entity_labels = labels
                out = outputs
                out = out.data.cpu().numpy()
                out = np.argmax(out, axis=1)
                out_len = len(out)               

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
                            num.append(int(i))
                            break
                        elif len(labels[j]) == 1:
                            num.append(1)
                original_labels = pad_sequences([[l for l in lab] for lab in original_labels],
                    maxlen=max_num, value=params.pad_tag_num, padding="post",  
                    dtype="long", truncating="post")

                luke_labels = [[params.num_of_tags for i in range(max_num)] for j in range(len(original_labels))]

                predicted_sequences = []
                for i in range(len(original_labels)):
                    predicted_sequence = ["O"] * len(original_labels[i])
                    predicted_sequences.append(predicted_sequence)

                idx = 0
                for i in range(len(original_labels)):
                    for j in range(max_num):
                        for k in range(j, max_num):
                            if j < num[i] and k < num[i]:
                                entity_val = id2val_entity[out[idx]] if out[idx] < 5 else "NIL"

                                if entity_val == "NIL":
                                    idx += 1
                                    continue
                                else:
                                    idx += 1
                                    predicted_sequences[i][j] = "B-" + entity_val
                                    if k > j:
                                        for l in range(j+1, k+1):
                                            predicted_sequences[i][l] = "I-" + entity_val
                            else:
                                idx += 1
                    for j in range(len(original_labels[i])):
                        if original_labels[i][j] != params.pad_tag_num:
                            luke_labels[i][j] = val2id[predicted_sequences[i][j]]

                luke_labels = torch.LongTensor(luke_labels)
                # if params.cuda:
                #     luke_labels = luke_labels.cuda()
                luke_labels = luke_labels.to(device=params.device)

                outputs = luke_labels.ravel()
                labels = original_labels

            #labels = torch.LongTensor(labels)
            labels = torch.LongTensor(labels)
            #if params.cuda:
            labels = labels.to(device=params.device)

            if params.we_method.lower() == 'luke':
                max_num = max([len(l) for l in entity_labels])
                entity_labels = pad_sequences([[l for l in lab] for lab in entity_labels],
                    maxlen=max_num, value=params.pad_tag_num, padding="post",
                    dtype="long", truncating="post")
                entity_labels = torch.LongTensor(entity_labels)
                #if params.cuda:
                entity_labels = entity_labels.to(device=params.device)
                loss = criterion(entity_outputs, entity_labels)
            else:
                loss = criterion(outputs, labels)

            outputs = outputs.data.cpu().numpy()
            labels = labels.data.cpu().numpy()

            total_loss += loss.item()
            acc, f1, labels_vals, outputs_vals = stats(outputs, labels, show_table = show_table)
            total_acc += acc
            total_f1_score += f1
            all_outputs.append(outputs_vals)
            all_labels.append(labels_vals) 

    flat_labels = [item for sublist in all_labels for item in sublist]
    flat_outputs = [item for sublist in all_outputs for item in sublist]
    print("\nEvaluation table: \n\n", seqeval.metrics.classification_report([flat_labels], [flat_outputs], digits=4))

    avg_loss = total_loss / num_batches
    avg_acc = total_acc / num_batches
    avg_f1_score = total_f1_score / num_batches

    return avg_loss, avg_acc, avg_f1_score


def train(model, optimizer, criterion, data_train_iterator, num_batches, params, id2val_entity, val2id, id2val):

    model.train()
    total_loss = 0.0
    total_acc = 0.0
    total_f1_score = 0.0
    all_outputs = []
    all_labels = []

    batches = trange(num_batches)
    for batch in batches:

        sentences, labels, contexts = next(data_train_iterator)
        outputs, labels = model(sentences, labels, contexts)            # W modelu padding labels do najdluzszego zdania w batch.

        #print("\n\n", outputs, "\n\n", labels, "\n")
        #quit()
        # ##



        if params.we_method.lower() == 'luke':                      # odczytanie labeli encji
            entity_outputs = outputs
            entity_labels = labels
            out = outputs
            out = out.data.cpu().numpy()
            out = np.argmax(out, axis=1)
            out_len = len(out)               

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
                        num.append(int(i))
                        break
                    elif len(labels[j]) == 1:
                        num.append(1)
            original_labels = pad_sequences([[l for l in lab] for lab in original_labels],
                maxlen=max_num, value=params.pad_tag_num, padding="post",  
                dtype="long", truncating="post")

            luke_labels = [[params.num_of_tags for i in range(max_num)] for j in range(len(original_labels))]

            predicted_sequences = []
            for i in range(len(original_labels)):
                predicted_sequence = ["O"] * len(original_labels[i])
                predicted_sequences.append(predicted_sequence)

            idx = 0
            for i in range(len(original_labels)):
                for j in range(max_num):
                    for k in range(j, max_num):
                        if j < num[i] and k < num[i]:
                            entity_val = id2val_entity[out[idx]] if out[idx] < 5 else "NIL"

                            if entity_val == "NIL":
                                idx += 1
                                continue
                            else:
                                idx += 1
                                predicted_sequences[i][j] = "B-" + entity_val
                                if k > j:
                                    for l in range(j+1, k+1):
                                        predicted_sequences[i][l] = "I-" + entity_val
                        else:
                            idx += 1
                for j in range(len(original_labels[i])):
                    if original_labels[i][j] != params.pad_tag_num:
                        luke_labels[i][j] = val2id[predicted_sequences[i][j]]

            luke_labels = torch.LongTensor(luke_labels)
            luke_labels = luke_labels.to(device=params.device)

            outputs = luke_labels.ravel()
            labels = original_labels

        labels = torch.LongTensor(labels)
        labels = labels.to(device=params.device)

        if params.we_method.lower() == 'luke':
            max_num = max([len(l) for l in entity_labels])
            entity_labels = pad_sequences([[l for l in lab] for lab in entity_labels],
                maxlen=max_num, value=params.pad_tag_num, padding="post",
                dtype="long", truncating="post")
            entity_labels = torch.LongTensor(entity_labels)
            #if params.cuda:
            entity_labels = entity_labels.to(device=params.device)
            loss = criterion(entity_outputs, entity_labels)
        else:
            loss = criterion(outputs, labels)
        
        optimizer.zero_grad()
        loss.backward()

        optimizer.step()

        outputs = outputs.data.cpu().numpy()
        labels = labels.data.cpu().numpy()

        total_loss += loss.item()
        if params.we_method.lower() == 'luke':
            total_acc += accuracy_luke(outputs, labels)
        else:
            acc, f1, labels_vals, outputs_vals = stats(outputs, labels)
            total_acc += acc
            total_f1_score += f1
            all_outputs.append(outputs_vals)
            all_labels.append(labels_vals) 

        batches.set_postfix(accuracy='{:05.3f}'.format(total_acc/(batch+1)), loss='{:05.3f}'.format(total_loss/(batch+1)))

    flat_labels = [item for sublist in all_labels for item in sublist]
    flat_outputs = [item for sublist in all_outputs for item in sublist]
    print("\nTraining table: \n\n", seqeval.metrics.classification_report([flat_labels], [flat_outputs], digits=4))
    avg_loss = total_loss / num_batches
    avg_acc = total_acc / num_batches
    avg_f1_score = total_f1_score / num_batches

    return avg_loss, avg_acc, avg_f1_score



if __name__ == '__main__':
    print("train start")
    params = params.Params()

    torch.manual_seed(params.seed)
    #params.cuda = torch.cuda.is_available()
    if torch.cuda.is_available():
        params.device = torch.device('cuda')
        torch.cuda.manual_seed(params.seed)
    elif torch.backends.mps.is_available():
        params.device = torch.device('mps')

    # Getting data:
    dataset_loader = DatasetLoader(params)
    id2val = dataset_loader.id2val
    val2id = dataset_loader.val2id
    id2val_entity = dataset_loader.id2val_entity
    val2id_entity = dataset_loader.val2id_entity
    data_train = dataset_loader.load_data("train", params)
    data_val = dataset_loader.load_data("val", params)

    #model = Model(params, id2val, val2id, val2id_entity).cuda() if params.cuda else Model(params, id2val, val2id, val2id_entity)
    model = Model(params, id2val, val2id, val2id_entity).to(device=params.device)
    optimizer = optim.Adam(model.parameters(), lr=params.learning_rate)
 
    criterion = loss_fun    

    print("\n\nTraining...")
    best_acc = -1.0
    best_epoch = -1
    best_epoch_loss = -1.0
    total_time = 0.0

    best_f1_score = -1.0
    best_f1_epoch = -1
    
    for epoch in range(params.num_epochs):
        print("Epoch {}/{}".format(epoch + 1, params.num_epochs), )

        start_train_time = time.time()
        # Training:
        num_batches = (params.train_size + 1) // params.train_batch_size           # number of batches in one epoch
        data_train_iterator = dataset_loader.data_iterator(data_train, params.train_size, params.train_batch_size, params, shuffle=True)
        avg_loss, avg_acc, avg_f1_score = train(model, optimizer, criterion, data_train_iterator, num_batches, params, id2val_entity, val2id, id2val)

        end_train_time = time.time()
        train_time = end_train_time - start_train_time
        print("Average train loss: ", avg_loss)
        print("Average train accuracy: ", avg_acc)
        print("Average train f1 score: ", avg_f1_score)
        print("Training time: ", train_time, "\n")
        total_time += train_time

        # Validation:
        num_batches = (params.val_size + 1) // params.val_batch_size
        data_val_iterator = dataset_loader.data_iterator(data_val, params.val_size, params.val_batch_size, params, shuffle=False)
        avg_loss, avg_acc, avg_f1_score = evaluate(model, criterion, data_val_iterator, num_batches, params, id2val_entity)

        end_val_time = time.time()
        val_time = end_val_time - end_train_time
        print("Average val loss: ", avg_loss)
        print("Average val accuracy: ", avg_acc)
        print("Average val f1 score: ", avg_f1_score)
        print("Validation time: ", val_time, "\n\n")
        total_time += val_time

        if avg_acc > best_acc:
            best_acc = avg_acc
            best_epoch = epoch
            best_epoch_loss = avg_loss
        
        if avg_f1_score > best_f1_score:
            best_f1_score = avg_f1_score
            best_f1_epoch = epoch

    print("\nBest accuracy: {:05.3f} for epoch number {} with the loss: {:05.3f}".format(best_acc, best_epoch+1, best_epoch_loss))
    print("\nBest f1 score: {:05.3f} for epoch number {}".format(best_f1_score, best_f1_epoch))

    
    print("\n\nTesting...")
    data_test = dataset_loader.load_data("test", params)

    start_test_time = time.time()
    num_batches = (params.test_size + 1) // params.val_batch_size
    data_test_iterator = dataset_loader.data_iterator(data_test, params.test_size, params.val_batch_size, params, shuffle=False)
    avg_loss, avg_acc, avg_f1_score = evaluate(model, criterion, data_test_iterator, num_batches, params, id2val_entity, show_table = True)
    end_test_time = time.time()
    test_time = start_test_time - end_test_time
    total_time += test_time

    print("Test accuracy: {:05.3f} with the loss: {:05.3f}".format(avg_acc, avg_loss))
    print("Test f1 score: {:05.3f}".format(avg_f1_score))
    print("Total time: {:05.3f}". format(total_time))