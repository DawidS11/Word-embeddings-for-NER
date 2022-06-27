from transformers import BertTokenizer, RobertaTokenizer, LukeTokenizer

# bert_tokenizer = BertTokenizer.from_pretrained("bert-base-cased")
# roberta_tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
# luke_tokenizer = LukeTokenizer.from_pretrained("studio-ousia/luke-base")

# def get_context_kaggle(dataset, params):
    # '''
    # dataset - list of lists of words
    # '''

    # if params.we_method.lower() == 'bert':
    #     tokenizer = BertTokenizer.from_pretrained("bert-base-cased")
    # elif params.we_method.lower() == 'roberta':
    #     tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
    # elif params.we_method.lower() == 'luke':
    #     tokenizer = LukeTokenizer.from_pretrained("studio-ousia/luke-base")
    
    # dataset_flat = sum(dataset, [])      # list of words
    
    # sentence_beg = 0     # indeks pierwszego słowa w zdaniu
    # sentence_end = 0     # indeks ostatniego słowa w zdaniu
    # for sen in dataset:
    #     sentence_end += len(sen)

    #     subword_lengths = [len(tokenizer.tokenize(w)) for w in sen]
    #     total_subword_length = sum(subword_lengths)

    #     if total_subword_length <= params.max_context_len:
    #         context_beg = sentence_beg
    #         context_end = sentence_end

    #     sentence_beg += (sentence_end + 1)




def get_context_conll2003(documents, params, val2id):
    
    if params.we_method.lower() == 'bert' or params.we_method.lower() == 'glove':
        tokenizer = BertTokenizer.from_pretrained("bert-base-cased")
    elif params.we_method.lower() == 'roberta':
        tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
    elif params.we_method.lower() == 'luke':
        tokenizer = LukeTokenizer.from_pretrained("studio-ousia/luke-base")
    
    contexts = []

    for document in documents:
        sentences_flat = sum(document["sentences"], [])
        labels_flat = sum(document["sentences_labels"], [])
        text_labels = [val2id[lab] for lab in labels_flat]
        subword_lengths = [len(tokenizer.tokenize(w)) for w in sentences_flat]
        total_subword_length = sum(subword_lengths)

        sentence_beg = 0     
        sentence_end = 0
        for sen, lab in zip(document["sentences"], document["sentences_labels"]):
            lab = [val2id[l] for l in lab]
            sentence_end += len(sen)
            
            if total_subword_length <= params.max_context_len:
                if sentence_beg != sentence_end:
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
                if context_beg != context_end:
                    contexts.append(dict(
                        sentence=sen,
                        labels=lab,
                        context_text=sentences_flat[context_beg:context_end],
                        context_labels=text_labels[context_beg:context_end],
                        sentence_beg=sentence_beg-context_beg,
                        sentence_end=sentence_end-context_beg,
                    ))

            sentence_beg += len(sen)

    return contexts