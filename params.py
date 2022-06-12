
class Params():
    def __init__(self):

        self.train_dataset_size = 0.7
        self.eval_dataset_size = 0.15
        self.test_dataset_size = 0.15        # 1 - train_dataset_size - val_dataset_size

        self.wb_method = 'luke' # glove / elmo / bert / roberta / luke
        self.nn_method = 'lstm' # rnn / lstm / cnn

        self.cuda = False    
        self.seed = 2022  

        self.num_epochs = 5
        self.learning_rate = 1e-3
        self.batch_size = 32

        self.embedding_dim = 100     
        self.hidden_dim = 100   
        self.glove_dim = 50
        self.elmo_dim = 512
        self.bert_dim = 1024 
        self.roberta_dim = 1024       
        self.luke_dim = 1024                 

        self.elmo_options_file = "./data/elmo/elmo_2x2048_256_2048cnn_1xhighway_options.json"
        self.elmo_weight_file = "./data/elmo/elmo_2x2048_256_2048cnn_1xhighway_weights.hdf5"

        self.data_dir = 'data/kaggle/'  
        self.glove_dir = 'data/glove/'  
        # self.data_dir = '/content/'  
        # self.glove_dir = '/content/' 

        self.train_size = 0
        self.eval_size = 0
        self.test_size = 0
        self.num_of_tags  = 0
        self.max_sen_len = 0
        self.vocab_size = 0

        self.pad_word = 'PAD'
        self.pad_tag = 'O'
        self.pad_tag_num = -1
        self.unk_word = 'UNK'