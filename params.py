


class Params():
    def __init__(self):

        self.train_dataset_size = 0.7
        self.val_dataset_size = 0.15
        self.test_dataset_size = 0.15        # 1 - train_dataset_size - val_dataset_size

        self.wb_method = 'bert'
        self.nn_method = 'bi-lstm'

        self.num_epochs = 1

        self.learning_rate = 1e-3
        self.batch_size = 5
        self.embedding_dim = 100     
        self.lstm_hidden_dim = 100   

        self.train_size = 0
        self.val_size = 0
        self.test_size = 0
        self.num_of_tags  = 0
        self.max_sen_len = 0

        self.cuda = False                       

        self.PAD_TAG = 'O'