#!/usr/bin/env python3
"""Functions for putting examples into torch format."""

from collections import Counter
import torch


def vectorize(ex, model, single_answer=False):
    """Torchify a single example."""
    args = model.args
    word_dict = model.word_dict
    feature_dict = model.feature_dict

    # Index words
    document = torch.LongTensor([word_dict[w] for w in ex['document']])
    question = torch.LongTensor([word_dict[w] for w in ex['question']])

    if args.char_embedding:
        # Index characters
        char_dict = model.char_dict
        maxwordlen = args.maxwordlen
        document_char = torch.LongTensor([[char_dict[ch] for ch in w[:maxwordlen]] +
                                          [char_dict['<NULLC>']]*(maxwordlen - len(w[:maxwordlen])) for w in ex['document']])
        question_char = torch.LongTensor([[char_dict[ch] for ch in w[:maxwordlen]] +
                                          [char_dict['<NULLC>']]*(maxwordlen - len(w[:maxwordlen])) for w in ex['question']])

    features = torch.zeros(len(ex['question']), 2)
    wh_words = ['what', 'who', 'how', 'when', 'which', 'where', 'why']
    qwords = ex['question']
    first_idx = -1
    for word_idx in range(len(qwords)):
        if qwords[word_idx].lower() in wh_words:
            first_idx = word_idx
            break
    if first_idx == -1:
        first_idx = 0
    if len(qwords) < 2:
        assert first_idx == 0
        second_idx = 0
    elif first_idx == len(qwords) - 1:
        second_idx = first_idx - 1
    else:
        second_idx = first_idx + 1
    features[first_idx][0] = 1.0
    features[second_idx][1] = 1.0

    if 'answers' not in ex:
        if args.char_embedding:
            return document, features, document_char, question, question_char, ex['id']
        else:
            return document, features, question, ex['id']

    # ...or with target(s) (might still be empty if answers is empty)
    if single_answer:
        assert(len(ex['answers']) > 0)
        start = torch.LongTensor(1).fill_(ex['answers'][0][0])
        end = torch.LongTensor(1).fill_(ex['answers'][0][1])
    else:
        start = [a[0] for a in ex['answers']]
        end = [a[1] for a in ex['answers']]

    if args.char_embedding:
        return document, features, document_char, question, question_char, start, end, ex['id']
    else:
        return document, features, question, start, end, ex['id']


def batchify(batch):
    """Gather a batch of individual examples into one batch."""
    NUM_INPUTS = 3
    NUM_TARGETS = 2
    NUM_EXTRA = 1

    ids = [ex[-1] for ex in batch]
    docs = [ex[0] for ex in batch]
    features = [ex[1] for ex in batch]
    questions = [ex[2] for ex in batch]

    # Batch documents and features
    max_length = max([d.size(0) for d in docs])
    max_qn_length = max([q.size(0) for q in questions])
    x1 = torch.LongTensor(len(docs), max_length).zero_()
    x1_mask = torch.ByteTensor(len(docs), max_length).fill_(1)
    if features[0] is None:
        x1_f = None
    else:
        x1_f = torch.zeros(len(questions), max_qn_length, features[0].size(1))
    for i, d in enumerate(docs):
        x1[i, :d.size(0)].copy_(d)
        x1_mask[i, :d.size(0)].fill_(0)

    # Batch questions
    max_length = max([q.size(0) for q in questions])
    x2 = torch.LongTensor(len(questions), max_length).zero_()
    x2_mask = torch.ByteTensor(len(questions), max_length).fill_(1)
    for i, q in enumerate(questions):
        x2[i, :q.size(0)].copy_(q)
        x2_mask[i, :q.size(0)].fill_(0)
        if x1_f is not None:
            x1_f[i, :q.size(0)].copy_(features[i])

    # Maybe return without targets
    if len(batch[0]) == NUM_INPUTS + NUM_EXTRA:
        return x1, x1_f, x1_mask, x2, x2_mask, ids

    elif len(batch[0]) == NUM_INPUTS + NUM_EXTRA + NUM_TARGETS:
        # ...Otherwise add targets
        if torch.is_tensor(batch[0][3]):
            y_s = torch.cat([ex[3] for ex in batch])
            y_e = torch.cat([ex[4] for ex in batch])
        else:
            y_s = [ex[3] for ex in batch]
            y_e = [ex[4] for ex in batch]
    else:
        raise RuntimeError('Incorrect number of inputs per example.')

    return x1, x1_f, x1_mask, x2, x2_mask, y_s, y_e, ids


def batchify_with_charemb(batch):
    """Gather a batch of individual examples into one batch."""
    NUM_INPUTS = 5
    NUM_TARGETS = 2
    NUM_EXTRA = 1

    ids = [ex[-1] for ex in batch]
    docs = [ex[0] for ex in batch]
    features = [ex[1] for ex in batch]
    docs_chars = [ex[2] for ex in batch]
    questions = [ex[3] for ex in batch]
    questions_chars = [ex[4] for ex in batch]

    # maxwordlen = docs_chars.size(2)
    maxwordlen = len(docs_chars[0][1])

    # Batch documents and features
    max_length = max([d.size(0) for d in docs])
    max_qn_length = max([q.size(0) for q in questions])
    x1 = torch.LongTensor(len(docs), max_length).zero_()
    x1_char = torch.LongTensor(len(docs), max_length, maxwordlen).zero_()
    x1_mask = torch.ByteTensor(len(docs), max_length).fill_(1)
    if features[0] is None:
        x1_f = None
    else:
        x1_f = torch.zeros(len(questions), max_qn_length, features[0].size(1))
    for i, d in enumerate(docs):
        x1[i, :d.size(0)].copy_(d)
        x1_char[i, :d.size(0), :maxwordlen].copy_(docs_chars[i])
        x1_mask[i, :d.size(0)].fill_(0)

    # Batch questions
    max_length = max([q.size(0) for q in questions])
    x2 = torch.LongTensor(len(questions), max_length).zero_()
    x2_char = torch.LongTensor(len(questions), max_length, maxwordlen).zero_()
    x2_mask = torch.ByteTensor(len(questions), max_length).fill_(1)
    for i, q in enumerate(questions):
        x2[i, :q.size(0)].copy_(q)
        x2_char[i, :q.size(0), :maxwordlen].copy_(questions_chars[i])
        x2_mask[i, :q.size(0)].fill_(0)
        if x1_f is not None:
            x1_f[i, :q.size(0)].copy_(features[i])

    # Maybe return without targets
    if len(batch[0]) == NUM_INPUTS + NUM_EXTRA:
        return x1, x1_f, x1_char, x1_mask, x2, x2_char, x2_mask, ids

    elif len(batch[0]) == NUM_INPUTS + NUM_EXTRA + NUM_TARGETS:
        # ...Otherwise add targets
        if torch.is_tensor(batch[0][5]):
            y_s = torch.cat([ex[5] for ex in batch])
            y_e = torch.cat([ex[6] for ex in batch])
        else:
            y_s = [ex[5] for ex in batch]
            y_e = [ex[6] for ex in batch]
    else:
        raise RuntimeError('Incorrect number of inputs per example.')

    return x1, x1_f, x1_char, x1_mask, x2, x2_char, x2_mask, y_s, y_e, ids
