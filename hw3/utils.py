import re
import torch
import numpy as np
from collections import Counter


def get_device(force_cpu, status=True):
    # if not force_cpu and torch.backends.mps.is_available():
    # 	device = torch.device('mps')
    # 	if status:
    # 		print("Using MPS")
    # elif not force_cpu and torch.cuda.is_available():
    if not force_cpu and torch.cuda.is_available():
        device = torch.device("cuda")
        if status:
            print("Using CUDA")
    else:
        device = torch.device("cpu")
        if status:
            print("Using CPU")
    return device


def preprocess_string(s):
    # Remove all non-word characters (everything except numbers and letters)
    s = re.sub(r"[^\w\s]", "", s)
    # Replace all runs of whitespaces with one space
    s = re.sub(r"\s+", " ", s)
    # replace digits with no space
    s = re.sub(r"\d", "", s)
    return s


def build_tokenizer_table(train, vocab_size=1000):
    word_list = []
    padded_lens = []
    inst_count = 0
    for episode in train:
        for inst, _ in episode:
            inst = preprocess_string(inst)
            padded_len = 2  # start/end
            for word in inst.lower().split():
                if len(word) > 0:
                    word_list.append(word)
                    padded_len += 1
            padded_lens.append(padded_len)
    corpus = Counter(word_list)
    corpus_ = sorted(corpus, key=corpus.get, reverse=True)[
        : vocab_size - 4
    ]  # save room for <pad>, <start>, <end>, and <unk>
    vocab_to_index = {w: i + 4 for i, w in enumerate(corpus_)}
    vocab_to_index["<pad>"] = 0
    vocab_to_index["<start>"] = 1
    vocab_to_index["<end>"] = 2
    vocab_to_index["<unk>"] = 3
    index_to_vocab = {vocab_to_index[w]: w for w in vocab_to_index}
    return (
        vocab_to_index,
        index_to_vocab,
        int(np.average(padded_lens) + np.std(padded_lens) * 2 + 0.5),
    )


def build_output_tables(train):
    actions = set()
    targets = set()
    for episode in train:
        for _, outseq in episode:
            a, t = outseq
            actions.add(a)
            targets.add(t)
    actions_to_index = {a: i + 3 for i, a in enumerate(actions)}
    targets_to_index = {t: i + 3 for i, t in enumerate(targets)}
    actions_to_index["<pad>"] = 0
    actions_to_index["<start>"] = 1
    actions_to_index["<end>"] = 2
    targets_to_index["<pad>"] = 0
    targets_to_index["<start>"] = 1
    targets_to_index["<end>"] = 2
    index_to_actions = {actions_to_index[a]: a for a in actions_to_index}
    index_to_targets = {targets_to_index[t]: t for t in targets_to_index}
    #print(actions_to_index,targets_to_index)
    return actions_to_index, index_to_actions, targets_to_index, index_to_targets

def prefix_match(predicted_labels, gt_labels):
    # predicted and gt are sequences of (action, target) labels, the sequences should be of same length
    # computes how many matching (action, target) labels there are between predicted and gt
    # is a number between 0 and 1 

    seq_length = len(gt_labels)
    
    for i in range(seq_length):
        if predicted_labels[i] != gt_labels[i]:
            break
    
    pm = (1.0 / seq_length) * i

    return pm

def custom_match(predicted_labels, gt_labels):
    # input sequences, then return the percent of each sequence that are same, with option to test only for the ones that are meaningful (disregard paddings after eos)
    # dumb implementation because my brain processor is fried to golden brown borderline overcooked to black
    # on that note, I think brain is actually on the menu for some cultures.
    # careful of prions though.

    seq_length = len(gt_labels)

    counter = 0
    #print("counting ",predicted_labels,gt_labels)
    for i in range(seq_length):
        if predicted_labels[i] == gt_labels[i]:
            counter+=1
        if(gt_labels[i] == 2): #end token
            break
    #print(counter)
    
    #print("testing val_acc: ", predicted_labels,gt_labels)
    percent = (0.5/i)*counter
    #print(percent)

    return percent
