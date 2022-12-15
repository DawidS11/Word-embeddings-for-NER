import warnings
warnings.filterwarnings('ignore')  # "error", "ignore", "always", "default", "module" or "once"

import os

os.environ['KMP_DUPLICATE_LIB_OK']='True'

import numpy as np
import torch
import torch.optim as optim
from tqdm import trange
import time

from model import Model
from dataset_loader import DatasetLoader
import sklearn.metrics
import seqeval.metrics
import params


def stats(outputs, labels):
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

    f1 = sklearn.metrics.f1_score(labels_not_masked, outputs_not_masked, average='micro')

    if my_params.we_method.lower() == 'luke':
        labels_vals = [id2val_entity[label] if label < my_params.num_of_tags_entity else 'NIL' for label in labels_not_masked]
        outputs_vals = [id2val_entity[output] if output < my_params.num_of_tags_entity else 'NIL' for output in outputs_not_masked]
    else:
        labels_vals = [id2val[label] if label < my_params.num_of_tags else 'O' for label in labels_not_masked]
        outputs_vals = [id2val[output] if output < my_params.num_of_tags else 'O' for output in outputs_not_masked]

    return accuracy, f1, labels_vals, outputs_vals


def loss_fun(outputs, labels):
    labels = labels.ravel()         # (1D: batch_size*seq_len)
    mask = (labels >= 0).float()
    
    #labels = labels % outputs.shape[1]     # nie dziala z device="mps"
    labels2 = []
    for l in labels:
        labels2.append(l.item() % outputs.shape[1])
    labels = torch.LongTensor(labels2)
    labels = labels.to(device=my_params.get_device())
    
    num_tokens = int(torch.sum(mask))
    
    return -torch.sum(outputs[range(outputs.shape[0]), labels]*mask)/num_tokens


def evaluate(model, criterion, data_eval_iterator, num_batches, show_table):
    
    model.eval()
    total_loss = 0.0
    total_acc = 0.0
    total_f1_score = 0.0
    all_outputs = []
    all_labels = []

    with torch.no_grad():
        batches = trange(num_batches)
        for batch in batches:

            sentences, labels, contexts = next(data_eval_iterator)
            outputs, labels = model(sentences, labels, contexts)

            labels = torch.LongTensor(labels)
            labels = labels.to(device=my_params.device)

            loss = criterion(outputs, labels)

            outputs = outputs.data.cpu().numpy()
            labels = labels.data.cpu().numpy()

            total_loss += loss.item()
            acc, f1, labels_vals, outputs_vals = stats(outputs, labels)
            total_acc += acc
            total_f1_score += f1
            all_outputs.append(outputs_vals)
            all_labels.append(labels_vals) 

    flat_labels = [item for sublist in all_labels for item in sublist]
    flat_outputs = [item for sublist in all_outputs for item in sublist]

    if show_table:
        if my_params.we_method.lower() == 'luke':
            x = []
            y = []
            for i in range(len(flat_labels)):
                if flat_labels[i] == 'NIL':
                    x.append('O')
                else:
                    x.append('B-' + flat_labels[i])

                if flat_outputs[i] == 'NIL':
                    y.append('O')
                else:
                    y.append('B-' + flat_outputs[i])
            
            print("\nEvaluation table: \n\n", seqeval.metrics.classification_report([x], [y], digits=4))
            print("Evaluation accuracy: ", seqeval.metrics.accuracy_score([x], [y]))
            print("Evaluation f1_score micro: ", seqeval.metrics.f1_score([x], [y]), "\n")

            # labels_without_NIL = [l for l in val2id.keys() if l != 'NIL']
            # print("\nEvaluation table: \n\n", sklearn.metrics.classification_report(flat_labels, flat_outputs, digits=4, labels=labels_without_NIL))
            # #print("\nEvaluation accuracy: ", sklearn.metrics.accuracy_score(flat_labels, flat_outputs, labels=labels_without_NIL))
            # print("Evaluation f1_score micro: ", sklearn.metrics.f1_score(flat_labels, flat_outputs, average='micro', labels=labels_without_NIL), "\n")
        else:
            labels_without_O = [l for l in val2id.keys() if l != 'O']
            print("\nEvaluation table: \n\n", sklearn.metrics.classification_report(flat_labels, flat_outputs, digits=4, labels=labels_without_O))
            #print("Evaluation accuracy: ", sklearn.metrics.accuracy_score(flat_labels, flat_outputs, labels=labels_without_O))
            print("Evaluation f1_score micro: ", sklearn.metrics.f1_score(flat_labels, flat_outputs, average='micro', labels=labels_without_O), "\n\n")
            print("\nEvaluation table: \n\n", seqeval.metrics.classification_report([flat_labels], [flat_outputs], digits=4))
            print("Evaluation accuracy: ", seqeval.metrics.accuracy_score([flat_labels], [flat_outputs]))
            print("Evaluation f1_score micro: ", seqeval.metrics.f1_score([flat_labels], [flat_outputs]), "\n")
       
    avg_loss = total_loss / num_batches
    avg_acc = total_acc / num_batches
    avg_f1_score = total_f1_score / num_batches

    return avg_loss, avg_acc, avg_f1_score


def train(model, optimizer, criterion, data_train_iterator, num_batches):

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

        labels = torch.LongTensor(labels)
        labels = labels.to(device=my_params.device)

        loss = criterion(outputs, labels)
        
        optimizer.zero_grad()
        loss.backward()

        optimizer.step()

        outputs = outputs.data.cpu().numpy()
        labels = labels.data.cpu().numpy()

        total_loss += loss.item()
        acc, f1, labels_vals, outputs_vals = stats(outputs, labels)
        total_acc += acc
        total_f1_score += f1
        all_outputs.append(outputs_vals)
        all_labels.append(labels_vals) 

        batches.set_postfix(accuracy='{:05.3f}'.format(total_acc/(batch+1)), loss='{:05.3f}'.format(total_loss/(batch+1)))

    # flat_labels = [item for sublist in all_labels for item in sublist]
    # flat_outputs = [item for sublist in all_outputs for item in sublist]
    # if my_params.we_method.lower() == 'luke':
    #     print("\nTraining table: \n\n", sklearn.metrics.classification_report(flat_labels, flat_outputs, digits=4))
    #     print("\nTraining accuracy: ", sklearn.metrics.accuracy_score(flat_labels, flat_outputs))
    #     print("Training f1_score micro: ", sklearn.metrics.f1_score(flat_labels, flat_outputs, average='micro'), "\n")
    # else:
    #     print("\nTraining table: \n\n", seqeval.metrics.classification_report([flat_labels], [flat_outputs], digits=4))
    #     print("\nTraining accuracy: ", seqeval.metrics.accuracy_score([flat_labels], [flat_outputs]))
    #     print("Training f1_score micro: ", seqeval.metrics.f1_score([flat_labels], [flat_outputs]), "\n")
        
    avg_loss = total_loss / num_batches
    avg_acc = total_acc / num_batches
    avg_f1_score = total_f1_score / num_batches

    return avg_loss, avg_acc, avg_f1_score



if __name__ == '__main__':
    print("train start")
    my_params = params.Params()

    torch.manual_seed(my_params.seed)
    if torch.cuda.is_available():
        my_params.device = torch.device('cuda')
        torch.cuda.manual_seed(my_params.seed)
    elif torch.backends.mps.is_available():
        my_params.device = torch.device('mps')

    # Getting data:
    dataset_loader = DatasetLoader(my_params)
    id2val = dataset_loader.id2val
    val2id = dataset_loader.val2id
    id2val_entity = dataset_loader.id2val_entity
    val2id_entity = dataset_loader.val2id_entity
    data_train = dataset_loader.load_data("train")
    data_val = dataset_loader.load_data("val")

    #model = Model(my_params, id2val, val2id, val2id_entity).cuda() if my_params.cuda else Model(my_params, id2val, val2id, val2id_entity)
    model = Model(my_params, id2val, val2id, val2id_entity).to(device=my_params.device)
    optimizer = optim.Adam(model.parameters(), lr=my_params.learning_rate)
 
    criterion = loss_fun    

    print("\n\nTraining...")
    best_acc = -1.0
    best_epoch = -1
    best_epoch_loss = -1.0
    total_time = 0.0

    best_f1_score = -1.0
    best_f1_epoch = -1
    
    for epoch in range(my_params.num_epochs):
        print("Epoch {}/{}".format(epoch + 1, my_params.num_epochs), )

        start_train_time = time.time()
        # Training:
        num_batches = (my_params.train_size + 1) // my_params.train_batch_size           # number of batches in one epoch
        data_train_iterator = dataset_loader.data_iterator(data_train, my_params.train_size, my_params.train_batch_size, my_params, shuffle=True)
        avg_loss, avg_acc, avg_f1_score = train(model, optimizer, criterion, data_train_iterator, num_batches)

        end_train_time = time.time()
        train_time = end_train_time - start_train_time
        print("Average train loss: ", avg_loss)
        print("Average train accuracy: ", avg_acc)
        print("Average train f1 score: ", avg_f1_score)
        print("Training time: ", train_time, "\n")
        total_time += train_time

        # Validation:
        num_batches = (my_params.val_size + 1) // my_params.val_batch_size
        data_val_iterator = dataset_loader.data_iterator(data_val, my_params.val_size, my_params.val_batch_size, my_params, shuffle=False)
        avg_loss, avg_acc, avg_f1_score = evaluate(model, criterion, data_val_iterator, num_batches, False)

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
    data_test = dataset_loader.load_data("test")

    start_test_time = time.time()
    num_batches = (my_params.test_size + 1) // my_params.val_batch_size
    data_test_iterator = dataset_loader.data_iterator(data_test, my_params.test_size, my_params.val_batch_size, my_params, shuffle=False)
    avg_loss, avg_acc, avg_f1_score = evaluate(model, criterion, data_test_iterator, num_batches, True)
    end_test_time = time.time()
    test_time = start_test_time - end_test_time
    total_time += test_time

    print("Test accuracy: {:05.3f} with the loss: {:05.3f}".format(avg_acc, avg_loss))
    print("Test f1 score: {:05.3f}".format(avg_f1_score))
    print("Total time: {:05.3f}". format(total_time))