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




def get_context_conll2003(documents, params):
    
    if params.we_method.lower() == 'bert':
        tokenizer = BertTokenizer.from_pretrained("bert-base-cased")
    elif params.we_method.lower() == 'roberta':
        tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
    elif params.we_method.lower() == 'luke':
        tokenizer = LukeTokenizer.from_pretrained("studio-ousia/luke-base")
    
    contexts = []

    for document in documents:
        sentences_flat = sum(document["sentences"], [])
        labels_flat = sum(document["sentences_labels"], [])
        subword_lengths = [len(tokenizer.tokenize(w)) for w in sentences_flat]
        total_subword_length = sum(subword_lengths)

        sentence_beg = 0     
        sentence_end = 0
        for sen, lab in zip(document["sentences"], document["sentences_labels"]):
            sentence_end += (len(sen) - 1)

            if total_subword_length <= params.max_context_len:
                contexts.append(dict(
                    sentence=sen,
                    labels=lab,
                    context_sentences=sentences_flat,
                    context_labels=labels_flat,
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
                contexts.append(dict(
                    sentence=sen,
                    labels=lab,
                    context_sentences=sentences_flat[context_beg:context_end],
                    context_labels=labels_flat[context_beg:context_end],
                ))

            sentence_beg += len(sen)
    
    return contexts