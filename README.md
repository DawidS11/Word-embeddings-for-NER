# Word-embeddings-for-NER


A program comparing the impact of word embeddings in a Named Entity Recognition (NER) task.
Models for embeddings in use: GloVe, ELMo, BERT, RoBERTa. 
Also, finetuned on CoNLL-2003 LUKE is used, to compare the results of entity-focused model. Note that the program does not use the embeddings created by LUKE, but predicted types of entities.

The model's nerual network part is a simple LSTM or Conv2d (depends on what we specify in params.py). It uses the embeddings created by mentioned models and learn to predict types of entities. 

GloVe's files: https://nlp.stanford.edu/projects/glove/ (glove.6B.zip)

ELMo's files: https://drive.google.com/file/d/1jklcip5p4I4wML5w0HDmvzWTvGY5cInG/view?usp=sharing

CoNLL-2003: https://www.kaggle.com/datasets/alaakhaled/conll003-englishversion

Kaggle dataset: https://www.kaggle.com/datasets/abhinavwalia95/entity-annotated-corpus
