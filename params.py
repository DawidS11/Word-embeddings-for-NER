
class Params():
    def __init__(self):

        self.train_dataset_size = 0.7
        self.val_dataset_size = 0.15
        self.test_dataset_size = 0.15       

        self.we_method = 'bert_large' # glove / elmo / bert_base / bert_large / roberta / luke
        self.nn_method = 'lstm' # rnn / lstm / cnn
        self.dataset_name = 'kaggle' # kaggle / conll2003 / kaggle_small

        self.device = 'cpu' 
        self.seed = 2022  

        self.num_epochs = 5
        self.learning_rate = 1e-3
        self.train_batch_size = 4
        self.val_batch_size = 2
        self.dropout = 0.3

        self.embedding_dim = 100     
        self.hidden_dim = 100   
        self.glove_dim = 50
        self.elmo_dim = 512
        self.bert_base_dim = 768 
        self.bert_large_dim = 1024
        self.roberta_dim = 768       
        self.luke_dim = 768                 

        # self.data_dir = 'data/conll2003/' # 'data/kaggle/' 'data/conll2003/'
        # self.glove_dir = 'data/glove/'  
        # self.elmo_dir = 'data/elmo/'
        # self.data_dir = '/content/'  
        # self.glove_dir = '/content/' 
        # self.elmo_dir = '/content/' 
        self.data_dir = '/notebooks/'  
        self.glove_dir = '/notebooks/'  
        self.elmo_dir = '/notebooks/' 

        self.elmo_options_file = "elmo_2x2048_256_2048cnn_1xhighway_options.json"
        self.elmo_weight_file = "elmo_2x2048_256_2048cnn_1xhighway_weights.hdf5"

        self.train_size = 0
        self.val_size = 0
        self.test_size = 0
        self.num_of_tags  = 0
        self.num_of_tags_entity = 0
        self.max_sen_len = 0
        self.max_entity_num = 0
        self.vocab_size = 0

        self.max_context_len = 510  # maksymalny dla bert i roberta to 512 (510 + znaki początku i końca)
        self.pad_word = 'P'
        self.pad_tag = 'O'
        self.pad_tag_num = -1
        self.unk_word = 'UNK'

    def get_device(self):
        return self.device