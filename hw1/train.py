import tqdm
import torch
import argparse
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader,Dataset
import torch.nn as nn
import torch.optim as optim
from model import AlfredLSTM
import matplotlib.pyplot as plt

from utils import (
    get_device,
    preprocess_string,
    build_tokenizer_table,
    build_output_tables,
)


def setup_dataloader(args):
    """
    return:
        - train_loader: torch.utils.data.Dataloader
        - val_loader: torch.utils.data.Dataloader
    """
    # ================== TODO: CODE HERE ================== #
    # Task: Load the training data from provided json file.
    # Perform some preprocessing to tokenize the natural
    # language instructions and labels. Split the data into
    # train set and validataion set and create respective
    # dataloaders.

    # Hint: use the helper functions provided in utils.py
    # ===================================================== #

    # read json file and convert to a single string
    json_file = open("CSCI499_NaturalLanguageforInteractiveAI\hw1\lang_to_sem_data.json","r")

    # parsing json string
    import json
    json_string = json.load(json_file)

    # Extract instruction and label from train and validation
    train_all_data = json_string["train"]
    val_all_data = json_string["valid_seen"]

    train_table = build_tokenizer_table(train_all_data,1000)
    val_table = train_table


    train_a2i, train_i2a, train_t2i, train_i2t = build_output_tables(train_all_data)
    val_a2i, val_i2a, val_t2i, val_i2t = build_output_tables(train_all_data)

    # use to store tokenize given instructions
    train_data = []
    val_data = []


    # tokenize and only saving id's instead of actual words.
    tokenized_length=60
    for train_list in train_all_data:
        for line in train_list:
            tki = []
            tkl = []
            # append start token
            tki.append(1)
            instrs = preprocess_string(line[0]).lower().split()
            targets = line[1]
            for word in instrs:
                # check if unkown
                if word not in train_table[0]:
                    tki.append(3)
                else:
                    tki.append(train_table[0][word])
            # tokenize labels
            tkl.append(train_a2i[targets[0]])
            tkl.append(train_t2i[targets[1]])
            # append padding
            while len(tki)<(tokenized_length-1):
                tki.append(0)
            # append end token
            tki.append(2)
            # append to dataset
            tkit = torch.IntTensor(tki)
            tklt = torch.IntTensor(tkl)
            train_data.append((tkit,tklt))
    
    for val_lists in val_all_data:
        for line in val_lists:
            vki = []
            vkl = []
            # append start token
            vki.append(1)
            instrs = preprocess_string(line[0]).lower().split()
            targets = line[1]
            for word in instrs:
                # check if unkown
                if word not in val_table[0]:
                    vki.append(3)
                else:
                    vki.append(val_table[0][word])
            # tokenize labels
            vkl.append(val_a2i[targets[0]])
            vkl.append(val_t2i[targets[1]])
            # append padding
            while len(vki)<(tokenized_length-1):
                vki.append(0)
            # append end token
            vki.append(2)
            # append to dataset
            vkit = torch.IntTensor(vki)
            vklt = torch.IntTensor(vkl)
            val_data.append((vkit,vklt))

    # checking length and output
    #print(len(train_data))
    #print(train_data[14056])
    #print(len(val_data))
    #print(val_data[7563])

    #cropping for easier testing
    train_data = train_data[0:5000]
    val_data = val_data[0:1000]

    # create data loader
    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=32, shuffle=True)

    #print(len(list(train_loader)))


    # Just throw every table out there in return, why not. (Probably will cause some memory issue, but I will check later)
    return train_loader, val_loader, (train_table, val_table, train_a2i, train_i2a, train_t2i, train_i2t,val_a2i, val_i2a, val_t2i, val_i2t)


def setup_model(args,maps,device):
    """
    return:
        - model: YourOwnModelClass
    """
    # ================== TODO: CODE HERE ================== #
    # Task: Initialize your model.
    # ===================================================== #

    # model contains 2 output heads, so the output size is inputted as a list of 2 scalars
    dicts = maps
    #print(len(dicts))

    embedding_dim = 128
    hidden_dim = 128
    model = AlfredLSTM(embedding_dim,hidden_dim,len(dicts[0][0]),[len(dicts[2]),len(dicts[4])])
    model.to(device)
    return model


def setup_optimizer(args, model):
    """
    return:
        - action_criterion: loss_fn
        - target_criterion: loss_fn
        - optimizer: torch.optim
    """
    # ================== TODO: CODE HERE ================== #
    # Task: Initialize the loss function for action predictions
    # and target predictions. Also initialize your optimizer.
    # ===================================================== #
    action_criterion = nn.NLLLoss()
    target_criterion = nn.NLLLoss()
    lr = 3e-5
    optimizer = optim.Adam(model.parameters(), lr=lr)

    return action_criterion, target_criterion, optimizer


def train_epoch(
    args,
    model,
    loader,
    optimizer,
    action_criterion,
    target_criterion,
    device,
    training=True,
):
    epoch_action_loss = 0.0
    epoch_target_loss = 0.0

    # keep track of the model predictions for computing accuracy
    action_preds = []
    target_preds = []
    action_labels = []
    target_labels = []
    #print("training epoch triggered")
    #print(len(list(loader)))

    # iterate over each batch in the dataloader
    # NOTE: you may have additional outputs from the loader __getitem__, you can modify this
    counter = 0
    for (inputs, labels) in loader:

        #print("training epoch start")
        #print(inputs)
        #print(labels[:,0].long())
        #print (counter/len(loader))
        counter+=1

        # put model inputs to device
        inputs, labels = inputs.to(device), labels.to(device)

        # calculate the loss and train accuracy and perform backprop
        # NOTE: feel free to change the parameters to the model forward pass here + outputs
        actions_out, targets_out = model(inputs, labels)
        #actions_out = torch.exp(actions_out)
        #targets_out = torch.exp(targets_out)

        #print ("action and target out got")

        # calculate the action and target prediction loss
        # NOTE: we assume that labels is a tensor of size Bx2 where labels[:, 0] is the
        # action label and labels[:, 1] is the target label
        #print(actions_out.shape)
        #print(actions_out.squeeze().shape)
        #print(labels[:,0].long().shape)
        action_loss = action_criterion(actions_out.squeeze(), labels[:, 0].long())
        target_loss = target_criterion(targets_out.squeeze(), labels[:, 1].long())

        loss = action_loss + target_loss

        # step optimizer and compute gradients during training
        if training:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # logging
        epoch_action_loss += action_loss.item()
        epoch_target_loss += target_loss.item()

        # take the prediction with the highest probability
        # NOTE: this could change depending on if you apply Sigmoid in your forward pass
        action_preds_ = actions_out.argmax(-1)
        target_preds_ = targets_out.argmax(-1)

        # aggregate the batch predictions + labels
        action_preds.extend(action_preds_.cpu().numpy())
        target_preds.extend(target_preds_.cpu().numpy())
        action_labels.extend(labels[:, 0].cpu().numpy())
        target_labels.extend(labels[:, 1].cpu().numpy())

    action_acc = accuracy_score(action_preds, action_labels)
    target_acc = accuracy_score(target_preds, target_labels)

    return epoch_action_loss, epoch_target_loss, action_acc, target_acc


def validate(
    args, model, loader, optimizer, action_criterion, target_criterion, device
):
    # set model to eval mode
    model.eval()

    # don't compute gradients
    with torch.no_grad():

        val_action_loss, val_target_loss, action_acc, target_acc = train_epoch(
            args,
            model,
            loader,
            optimizer,
            action_criterion,
            target_criterion,
            device,
            training=False,
        )

    return val_action_loss, val_target_loss, action_acc, target_acc


def train(args, model, loaders, optimizer, action_criterion, target_criterion, device):
    # Train model for a fixed number of epochs
    # In each epoch we compute loss on each sample in our dataset and update the model
    # weights via backpropagation
    model.train()

    tal = []
    ttl = []
    taa = []
    tta = []
    val = []
    vtl = []
    vaa = []
    vta = []

    for epoch in tqdm.tqdm(range(args.num_epochs)):
        #print("training started")
        # train single epoch
        # returns loss for action and target prediction and accuracy
        (
            train_action_loss,
            train_target_loss,
            train_action_acc,
            train_target_acc,
        ) = train_epoch(
            args,
            model,
            loaders["train"],
            optimizer,
            action_criterion,
            target_criterion,
            device,
        )

        # some logging
        print(
            f"train action loss : {train_action_loss} | train target loss: {train_target_loss}"
        )
        print(
            f"train action acc : {train_action_acc} | train target acc: {train_target_acc}"
        )

        # some registering
        tal.append(train_action_loss)
        ttl.append(train_target_loss)
        taa.append(train_action_acc)
        tta.append(train_target_acc)

        # run validation every so often
        # during eval, we run a forward pass through the model and compute
        # loss and accuracy but we don't update the model weights
        if epoch % args.val_every == 0:
            val_action_loss, val_target_loss, val_action_acc, val_target_acc = validate(
                args,
                model,
                loaders["val"],
                optimizer,
                action_criterion,
                target_criterion,
                device,
            )

            print(
                f"val action loss : {val_action_loss} | val target loss: {val_target_loss}"
            )
            print(
                f"val action acc : {val_action_acc} | val target accs: {val_target_acc}"
            )
            val.append(val_action_loss)
            vtl.append(val_target_loss)
            vaa.append(val_action_acc)
            vta.append(val_target_acc)

    # ================== TODO: CODE HERE ================== #
    # Task: Implement some code to keep track of the model training and
    # evaluation loss. Use the matplotlib library to plot
    # 4 figures for 1) training loss, 2) training accuracy,
    # 3) validation loss, 4) validation accuracy
    # ===================================================== #
    # some plotting
    trainepoch = range(0,len(tal))
    valepoch = range(0,len(val))
    allgraph, ax = plt.subplots(2,2)
    ax[0,0].plot(trainepoch,tal,label="Train Action Loss")
    ax[0,0].plot(trainepoch,ttl,label="Train Target Loss")
    ax[0,0].legend()
    ax[0,0].set_title("Training Loss")
    ax[0,1].plot(trainepoch,taa,label="Train Action Acc")
    ax[0,1].plot(trainepoch,tta,label="Train Target Acc")
    ax[0,1].legend()
    ax[0,1].set_title("Training Acc")
    ax[1,0].plot(valepoch,val,label="Validation Action Loss")
    ax[1,0].plot(valepoch,vtl,label="Validation Target Loss")
    ax[1,0].legend()
    ax[1,0].set_title("Validation Loss")
    ax[1,1].plot(valepoch,vaa,label="Validation Action Acc")
    ax[1,1].plot(valepoch,vta,label="Validation Target Acc")
    ax[1,1].legend()
    ax[1,1].set_title("Validation Acc")

    allgraph.suptitle("Statistics")
    allgraph.tight_layout()

    plt.savefig("./stats.pdf")



def main(args):
    device = get_device(args.force_cpu)

    # get dataloaders
    train_loader, val_loader, maps = setup_dataloader(args)
    loaders = {"train": train_loader, "val": val_loader}

    # build model
    model = setup_model(args, maps, device)
    print(model)

    # get optimizer and loss functions
    action_criterion, target_criterion, optimizer = setup_optimizer(args, model)

    if args.eval:
        val_action_loss, val_target_loss, val_action_acc, val_target_acc = validate(
            args,
            model,
            loaders["val"],
            optimizer,
            action_criterion,
            target_criterion,
            device,
        )
    else:
        train(
            args, model, loaders, optimizer, action_criterion, target_criterion, device
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_data_fn", type=str, help="data file")
    parser.add_argument(
        "--model_output_dir", type=str, help="where to save model outputs"
    )
    parser.add_argument(
        "--batch_size", type=int, default=32, help="size of each batch in loader"
    )
    parser.add_argument("--force_cpu", action="store_true", help="debug mode")
    parser.add_argument("--eval", action="store_true", help="run eval")
    parser.add_argument("--num_epochs",type=int, default=1000, help="number of training epochs")
    parser.add_argument(
        "--val_every", default=5, help="number of epochs between every eval loop"
    )

    # ================== TODO: CODE HERE ================== #
    # Task (optional): Add any additional command line
    # parameters you may need here
    # ===================================================== #
    args = parser.parse_args()

    main(args)
