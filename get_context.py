from transformers import BertTokenizer, RobertaTokenizer, LukeTokenizer

'''
Each sentence has its context. In case of Kaggle dataset, it is only that sentence since the dataset is not divided into documents.
'''

def get_context_kaggle(sentences, labels, val2id):

    contexts = []
    for sen, lab in zip(sentences, labels):
        sentence = " ".join(sen)
        labels = [val2id[l] for l in lab]
        contexts.append(dict(
            sentence=sen,
            labels=labels,
            context_text=sen,
            context_labels=labels,
            sentence_beg=0,
            sentence_end=len(sentence),
        ))
    return contexts


def get_context_conll2003(documents, params, val2id):
     
    if params.we_method.lower() == 'bert_base':
        if params.bert_cased:
            tokenizer = BertTokenizer.from_pretrained("bert-base-cased")
        else:
            tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        params.max_context_len = 510
    elif params.we_method.lower() == 'bert_large':
        if params.bert_cased:
            tokenizer = BertTokenizer.from_pretrained("bert-large-cased")
        else:
            tokenizer = BertTokenizer.from_pretrained("bert-large-uncased")
        params.max_context_len = 510
    elif params.we_method.lower() == 'roberta':
        tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
        params.max_context_len = 510
    else:
        params.max_context_len = 2048                                              
    
    contexts = []
    
    for document in documents:
        sentences_flat = sum(document["sentences"], [])
        labels_flat = sum(document["sentences_labels"], [])
        text_labels = [val2id[lab] for lab in labels_flat]
        if params.we_method.lower() == 'glove' or params.we_method.lower() == 'elmo':
            subword_lengths = [len(w) for w in sentences_flat]                          # number of characters
        else:
            subword_lengths = [len(tokenizer.tokenize(w)) for w in sentences_flat]      # number of tokens
        total_subword_length = sum(subword_lengths)

        sentence_beg = 0     
        sentence_end = 0
        for sen, lab in zip(document["sentences"], document["sentences_labels"]):
            lab = [val2id[l] for l in lab]
            sentence_end += len(sen)                        
            if not sen:
                continue
            if total_subword_length <= params.max_context_len:
                if sentence_beg < sentence_end:
                    contexts.append(dict(
                        sentence=sen,
                        labels=lab,
                        context_text=sentences_flat,
                        context_labels=text_labels,
                        sentence_beg=sentence_beg,
                        sentence_end=sentence_end,
                    ))

            else:
                context_beg = sentence_beg
                context_end = sentence_end
                cur_length = sum(subword_lengths[context_beg:context_end])

                while True:
                    if context_beg > 0:
                        if cur_length + subword_lengths[context_beg - 1] <= params.max_context_len:
                            cur_length += subword_lengths[context_beg - 1]
                            context_beg -= 1
                        else:
                            break
                    if context_end < len(sentences_flat):
                        if cur_length + subword_lengths[context_end] <= params.max_context_len:
                            cur_length += subword_lengths[context_end]
                            context_end += 1
                        else:
                            break
                if context_beg < context_end:
                    contexts.append(dict(
                        sentence=sen,
                        labels=lab,
                        context_text=sentences_flat[context_beg:context_end],
                        context_labels=text_labels[context_beg:context_end],
                        sentence_beg=sentence_beg-context_beg,                  # -context_beg in order to point the beginning of the sentence in context_text
                        sentence_end=sentence_end-context_beg,                   
                    ))

            sentence_beg += len(sen)
            
    return contexts