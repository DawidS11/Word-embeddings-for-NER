import os
import numpy as np
from collections import Counter

def update_vocab(dataset, words):        
    for sen in dataset:
        words.update(sen)

def create_vocab(train_sentences, val_sentences, test_sentences, params):
    words = Counter()
    update_vocab(train_sentences, words)
    update_vocab(val_sentences, words)
    update_vocab(test_sentences, words)

    words = [tok for tok, count in words.items() if count >= 1]

    if params.pad_word not in words: words.append(params.pad_word)

    words.append(params.unk_word)

    params.vocab_size = len(words)

    # Saving words:
    with open(os.path.join(params.data_dir, 'words.txt'), "w") as f:
        for word in words:
            f.write(word + '\n')


def get_glove(params):
    vocab = {j.strip(): i for i, j in enumerate(open(os.path.join(params.data_dir, 'words.txt')), 0)}
    id2word = {vocab[i]: i for i in vocab}

    dim = 0
    w2v = {}
    for line in open(os.path.join(params.glove_dir, 'glove.6B.{}d.txt'.format(params.glove_dim))):
        line = line.strip().split()
        word = line[0]
        vec = list(map(float, line[1:]))
        dim = len(vec)
        w2v[word] = vec

    vecs = []
    vecs.append(np.random.uniform(low=-1.0, high=1.0, size=dim))

    for i in range(1, len(vocab) - 1):
        if id2word[i] in w2v:
            vecs.append(w2v[id2word[i]])
        else:
            vecs.append(vecs[0])
    vecs.append(np.zeros(dim))
    assert(len(vecs) == len(vocab))

    np.save(os.path.join(params.glove_dir, 'glove_{}d.npy'.format(dim)), np.array(vecs, dtype=np.float32))
    np.save(os.path.join(params.glove_dir, 'word2id.npy'), vocab)
    np.save(os.path.join(params.glove_dir, 'id2word.npy'), id2word)