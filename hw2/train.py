import argparse
import os
import tqdm
import torch
from sklearn.metrics import accuracy_score
import torch.nn as nn

from eval_utils import downstream_validation
import utils
import data_utils
from model import CBOW
import random
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

def setup_dataloader(args):
    """
    return:
        - train_loader: torch.utils.data.Dataloader
        - val_loader: torch.utils.data.Dataloader
    """

    # read in training data from books dataset
    sentences = data_utils.process_book_dir(args.data_dir)

    # build one hot maps for input and output
    (
        vocab_to_index,
        index_to_vocab,
        suggested_padding_len,
    ) = data_utils.build_tokenizer_table(sentences, vocab_size=args.vocab_size)

    # create encoded input and output numpy matrices for the entire dataset and then put them into tensors
    encoded_sentences, lens = data_utils.encode_data(
        sentences,
        vocab_to_index,
        suggested_padding_len,
    )

    # ================== TODO: CODE HERE ================== #
    # Task: Given the tokenized and encoded text, you need to
    # create inputs to the LM model you want to train.
    # E.g., could be target word in -> context out or
    # context in -> target word out.
    # You can build up that input/output table across all
    # encoded sentences in the dataset!
    # Then, split the data into train set and validation set
    # (you can use utils functions) and create respective
    # dataloaders.
    # ===================================================== #

    print(len(sentences))
    #print(encoded_sentences[10000])

    # split train sentences and val sentences
    random.shuffle(encoded_sentences)
    train_sentences_len = int(len(encoded_sentences)*0.7)
    train_sentences = encoded_sentences[:train_sentences_len]
    val_sentences = encoded_sentences[train_sentences_len:]

    # Context to target for each train and val
    train_C2T = []
    window = 6
    for sentence in train_sentences:
        # extract non-padded encoded sentence, since there is no need to use all the padding at the end for not going to process that tensor anyway
        temps = []
        for tk in sentence:
            if(tk!=0):
                temps.append(tk)
            else:
                break

        for i in range (window//2):
            temps.insert(0,0)
            temps.append(0)

        for index, word in enumerate(temps):
            temp = []

            # Construct pairing
            if index>=(window//2) and index<(len(temps)-window//2):
                temp.append(temps[index-3])
                temp.append(temps[index-2])
                temp.append(temps[index-1])
                temp.append(temps[index+1])
                temp.append(temps[index+2])
                temp.append(temps[index+3])
                # tensor conversion
                ttemp = torch.IntTensor(temp)
                train_C2T.append([ttemp,word])

    val_C2T = []
    for sentence in val_sentences:
        # extract non-padded encoded sentence, since there is no need to use all the padding at the end for not going to process that tensor anyway
        temps = []
        for tk in sentence:
            if(tk!=0):
                temps.append(tk)
            else:
                break

        for i in range (window//2):
            temps.insert(0,0)
            temps.append(0)

        for index, word in enumerate(temps):
            temp = []

            # Construct pairing
            if index>=(window//2) and index<(len(temps)-window//2):
                temp.append(temps[index-3])
                temp.append(temps[index-2])
                temp.append(temps[index-1])
                temp.append(temps[index+1])
                temp.append(temps[index+2])
                temp.append(temps[index+3])
                # tensor conversion
                ttemp = torch.IntTensor(temp)
                val_C2T.append([ttemp,word])

    # context to target word vector
    #C2T = []
    #window = 6
    #for sentence in encoded_sentences:
        # extract non-padded encoded sentence, since there is no need to use all the padding at the end for not going to process that tensor anyway
    #    temps = []
    #    for tk in sentence:
    #        if(tk!=0):
    #            temps.append(tk)
    #        else:
    #            break
    #
    #    for i in range (window//2):
    #        temps.insert(0,0)
    #        temps.append(0)
    #
    #    for index, word in enumerate(temps):
    #        temp = []
    #
            # Construct pairing
    #        if index>=(window//2) and index<(len(temps)-window//2):
    #            temp.append(temps[index-3])
    #            temp.append(temps[index-2])
    #            temp.append(temps[index-1])
    #            temp.append(temps[index+1])
    #            temp.append(temps[index+2])
    #            temp.append(temps[index+3])
    #            # tensor conversion
    #            ttemp = torch.IntTensor(temp)
    #            C2T.append([ttemp,word])
    
    #print(C2T)

    # random sample to notify finish
    print (len(train_C2T))
    print (train_C2T[40000])

    # random split train and val
    #random.shuffle(C2T)
    #train_len = int(len(C2T)*0.7)
    #train_set = C2T[:train_len]
    #val_set = C2T[train_len:]


    train_loader = DataLoader(train_C2T, batch_size=128, shuffle=True)
    val_loader = DataLoader(val_C2T, batch_size=128, shuffle=True)
    return train_loader, val_loader, index_to_vocab


def setup_model(args):
    """
    return:
        - model: YourOwnModelClass
    """
    # ================== TODO: CODE HERE ================== #
    # Task: Initialize your CBOW or Skip-Gram model.
    # ===================================================== #
    vocab_size = 3000
    model = CBOW(vocab_size)
    # forcing cuda
    model.cuda()
    return model


def setup_optimizer(args, model):
    """
    return:
        - criterion: loss_fn
        - optimizer: torch.optim
    """
    # ================== TODO: CODE HERE ================== #
    # Task: Initialize the loss function for predictions. 
    # Also initialize your optimizer.
    # ===================================================== #
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    return criterion, optimizer


def train_epoch(
    args,
    model,
    loader,
    optimizer,
    criterion,
    device,
    training=True,
):
    model.train()
    epoch_loss = 0.0

    # keep track of the model predictions for computing accuracy
    pred_labels = []
    target_labels = []

    # iterate over each batch in the dataloader
    # NOTE: you may have additional outputs from the loader __getitem__, you can modify this
    for (inputs, labels) in tqdm.tqdm(loader):
        # put model inputs to device
        inputs, labels = inputs.to(device).long(), labels.to(device).long()

        # calculate the loss and train accuracy and perform backprop
        # NOTE: feel free to change the parameters to the model forward pass here + outputs
        pred_logits = model(inputs)

        # calculate prediction loss
        loss = criterion(pred_logits.squeeze(), labels)

        # step optimizer and compute gradients during training
        if training:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # logging
        epoch_loss += loss.item()

        # compute metrics
        preds = pred_logits.argmax(-1)
        pred_labels.extend(preds.cpu().numpy())
        target_labels.extend(labels.cpu().numpy())

    acc = accuracy_score(pred_labels, target_labels)
    epoch_loss /= len(loader)

    return epoch_loss, acc


def validate(args, model, loader, optimizer, criterion, device):
    # set model to eval mode
    model.eval()

    # don't compute gradients
    with torch.no_grad():
        val_loss, val_acc = train_epoch(
            args,
            model,
            loader,
            optimizer,
            criterion,
            device,
            training=False,
        )

    return val_loss, val_acc


def main(args):

    print("starting")

    device = utils.get_device(args.force_cpu)

    # load analogies for downstream eval
    external_val_analogies = utils.read_analogies(args.analogies_fn)

    if args.downstream_eval:
        word_vec_file = os.path.join(args.output_dir, args.word_vector_fn)
        assert os.path.exists(word_vec_file), "need to train the word vecs first!"
        downstream_validation(word_vec_file, external_val_analogies)
        return

    # get dataloaders
    train_loader, val_loader, i2v = setup_dataloader(args)
    loaders = {"train": train_loader, "val": val_loader}

    # build model
    model = setup_model(args)
    # forcing cuda
    model.cuda()
    print(model)

    # get optimizer
    criterion, optimizer = setup_optimizer(args, model)

    # Store Information for graph
    tl = []
    ta = []
    vl = []
    va = []

    for epoch in range(args.num_epochs):
        # train model for a single epoch
        print(f"Epoch {epoch}")
        train_loss, train_acc = train_epoch(
            args,
            model,
            loaders["train"],
            optimizer,
            criterion,
            device,
        )

        print(f"train loss : {train_loss} | train acc: {train_acc}")

        # Logging training loss and accuracy
        tl.append(train_loss)
        ta.append(train_acc)

        if epoch % args.val_every == 0:
            val_loss, val_acc = validate(
                args,
                model,
                loaders["val"],
                optimizer,
                criterion,
                device,
            )
            print(f"val loss : {val_loss} | val acc: {val_acc}")

            # Logging validation loss and accuracy
            vl.append(val_loss)
            va.append(val_acc)

            # ======================= NOTE ======================== #
            # Saving the word vectors to disk and running the eval
            # can be costly when you do it multiple times. You could
            # change this to run only when your training has concluded.
            # However, incremental saving means if something crashes
            # later or you get bored and kill the process you'll still
            # have a word vector file and some results.
            # ===================================================== #

            # save word vectors
            word_vec_file = os.path.join(args.output_dir, args.word_vector_fn)
            print("saving word vec to ", word_vec_file)
            utils.save_word2vec_format(word_vec_file, model, i2v)

            # evaluate learned embeddings on a downstream task
            downstream_validation(word_vec_file, external_val_analogies)


        if epoch % args.save_every == 0:
            ckpt_file = os.path.join(args.output_dir, "model.ckpt")
            print("saving model to ", ckpt_file)
            torch.save(model, ckpt_file)
    
    # Drawing Graph
    trainepoch = range(0,len(tl))
    valepoch = range(0,len(vl))

    print(tl)
    print(ta)
    print(vl)
    print(va)
    print(trainepoch)
    print(valepoch)

    plt.plot(trainepoch, tl)
    plt.xlabel("training epoch")
    plt.ylabel("training loss")
    plt.title("training loss")
    plt.savefig("./statistic-temp/trainingloss.pdf")
    plt.clf()

    plt.plot(trainepoch, ta)
    plt.xlabel("training epoch")
    plt.ylabel("training accuracy")
    plt.title("training accuracy")
    plt.savefig("./statistic-temp/trainingacc.pdf")
    plt.clf()

    plt.plot(valepoch, vl)
    plt.xlabel("validation epoch")
    plt.ylabel("validation loss")
    plt.title("validation loss")
    plt.savefig("./statistic-temp/valloss.pdf")
    plt.clf()

    plt.plot(valepoch, va)
    plt.xlabel("validation epoch")
    plt.ylabel("validation accuracy")
    plt.title("validation accuracy")
    plt.savefig("./statistic-temp/valacc.pdf")
    plt.clf()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, help="where to save training outputs")
    parser.add_argument("--data_dir", type=str, help="where the book dataset is stored")
    parser.add_argument(
        "--downstream_eval",
        action="store_true",
        help="run downstream eval on trained word vecs",
    )
    # ======================= NOTE ======================== #
    # If you adjust the vocab_size down below 3000, there 
    # may be analogies in the downstream evaluation that have
    # words that are not in your vocabulary, resulting in
    # automatic (currently) zero score for an ABCD where one
    # of A, B, C, or D is not in the vocab. A visible warning
    # will be generated by the evaluation loop for these examples.
    # ===================================================== #
    parser.add_argument(
        "--vocab_size", type=int, default=3000, help="size of vocabulary"
    )
    parser.add_argument(
        "--batch_size", type=int, default=32, help="size of each batch in loader"
    )
    parser.add_argument("--force_cpu", action="store_true", help="debug mode")
    parser.add_argument(
        "--analogies_fn", type=str, help="filepath to the analogies json file"
    )
    parser.add_argument(
        "--word_vector_fn", type=str, help="filepath to store the learned word vectors",
        default='learned_word_vectors.txt'
    )
    parser.add_argument(
        "--num_epochs", default=30, type=int, help="number of training epochs"
    )
    parser.add_argument(
        "--val_every",
        default=5,
        type=int,
        help="number of epochs between every eval loop",
    )
    parser.add_argument(
        "--save_every",
        default=5,
        type=int,
        help="number of epochs between saving model checkpoint",
    )
    # ================== TODO: CODE HERE ================== #
    # Task (optional): Add any additional command line
    # parameters you may need here
    # ===================================================== #

    args = parser.parse_args()
    main(args)
