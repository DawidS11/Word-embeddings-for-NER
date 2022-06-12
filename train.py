import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
from tqdm import trange
import time

from model import Model
from kaggle_dataset_loader import KaggleDatasetLoader #, MyDataset
from torch.utils.data import DataLoader

import params


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

    return np.sum(outputs == labels)/float(np.sum(mask))


def evaluate(model, criterion, data_eval_iterator, num_batches, params):
     
    model.eval()
    total_loss = 0.0
    total_acc = 0.0

    for batch in range(num_batches):

        sentences, labels = next(data_eval_iterator)

        outputs, labels = model(sentences, labels)
        #outputs = model(sentences)

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


def train(model, optimizer, criterion, data_train_iterator, num_batches, params):

    model.train()
    total_loss = 0.0
    total_acc = 0.0

    batches = trange(num_batches)
    for batch in batches:

        sentences, labels = next(data_train_iterator)

        outputs, labels = model(sentences, labels)
        #outputs = model(sentences)

        labels = torch.LongTensor(labels)
        if params.cuda:
            labels = labels.cuda()
        
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()

        optimizer.step()

        outputs = outputs.data.cpu().numpy()
        labels = labels.data.cpu().numpy()

        total_loss += loss.item()
        total_acc += accuracy(outputs, labels)

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
    dataset_loader = KaggleDatasetLoader(params)
    data_train = dataset_loader.load_data("train", params)
    data_eval = dataset_loader.load_data("eval", params)
    # data_train = MyDataset("train", data_loader, params)
    # data_eval = MyDataset("val", data_loader, params)

    model = Model(dataset_loader, params).cuda() if params.cuda else Model(dataset_loader, params)
    optimizer = optim.Adam(model.parameters(), lr=params.learning_rate)
 
    criterion = loss_fun
  
    best_acc = -1.0
    best_epoch = -1
    best_epoch_loss = -1.0
    total_time = 0.0
    
    
    for epoch in range(params.num_epochs):

        print("Epoch {}/{}".format(epoch + 1, params.num_epochs), )

        start_time = time.time()
        # Training:
        num_batches = (params.train_size + 1) // params.batch_size           # number of batches in one epoch
        data_train_iterator = dataset_loader.data_iterator(data_train, num_batches, params, shuffle=True)
        avg_loss, avg_acc = train(model, optimizer, criterion, data_train_iterator, num_batches, params)

        end_train_time = time.time()
        training_time = end_train_time - start_time
        print("Average train loss: ", avg_loss)
        print("Average train accuracy: ", avg_acc)
        print("Training time: ", training_time)


        # Evaluating:
        num_batches = (params.eval_size + 1) // params.batch_size
        data_eval_iterator = dataset_loader.data_iterator(data_eval, num_batches, params, shuffle=False)
        avg_loss, avg_acc = evaluate(model, criterion, data_eval_iterator, num_batches, params)

        end_eval_time = time.time()
        evaluating_time = end_eval_time - end_train_time
        print("\nAverage eval loss: ", avg_loss)
        print("Average eval accuracy: ", avg_acc)
        print("Evaluating time: ", time.time() - end_train_time, "\n\n")

        if avg_acc > best_acc:
            best_acc = avg_acc
            best_epoch = epoch
            best_epoch_loss = avg_loss

    print("\nBest accuracy: {:05.3f} for epoch number {} with the loss: {:05.3f}".format(best_acc, best_epoch+1, best_epoch_loss))