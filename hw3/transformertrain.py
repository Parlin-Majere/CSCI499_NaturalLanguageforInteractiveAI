from tqdm import tqdm
import torch
import argparse
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader
from utils import (
    get_device,
    preprocess_string,
    build_tokenizer_table,
    build_output_tables,
    prefix_match,
    custom_match
)
# use for basic encoder decoder
from transformermodel import Transformer
import matplotlib as plt


def setup_dataloader(args):
    """
    return:
        - train_loader: torch.utils.data.Dataloader
        - val_loader: torch.utils.data.Dataloader
    """
    #print(args.in_data_fn)
    # ================== TODO: CODE HERE ================== #
    # Task: Load the training data from provided json file.
    # Perform some preprocessing to tokenize the natural
    # language instructions and labels. Split the data into
    # train set and validataion set and create respective
    # dataloaders.

    # Hint: use the helper functions provided in utils.py
    # ===================================================== #
    
    # device
    if (args.force_cpu):
        device = "cpu"
    else:
        device = "cuda"

    # tokenization
    # read json file and convert to a single string
    json_file = open(args.in_data_fn,"r")

    # parsing json string
    import json
    json_string = json.load(json_file)

    # Extract instruction and label from train and validation
    train_all_string = json_string["train"]
    val_all_string = json_string["valid_seen"]

    # Empty list for storing tokenized inputs
    train_list = []
    val_list = []

    # tokenizer table
    v2i, i2v, pad_length = build_tokenizer_table(train_all_string)
    # output table
    a2i, i2a, t2i, i2t = build_output_tables(train_all_string)

    # padding length calculated as padding_length * len(episode) - (len(episode)-1)*2 
    # last part to get rid of the extra start and end tokens as they are not quite needed here.
    # very weirdly the number of instructions in training dataset are all 11 with validation being 12
    # don't know if it is going to cause a problem
    #train_pad_length = pad_length * 11
    #val_pad_length = pad_length * 12
    ins_pad_length = 380
    target_pad_length = 30
    action_pad_length = 30

    last_t_length = 0
    last_a_length = 0
    maxt = 0
    maxa = 0

    # for each episode, process strings, and then tokenize
    for episode in train_all_string:
        # each entry in an episode is processed individually
        ins = []
        actionl = []
        targetl = []

        # append start token
        ins.append(1)
        actionl.append(1)
        targetl.append(1)

        # since doing seq2seq, need to combine all single entries in episode into a single sequence
        # in the form of ["compounded_low_level instructions", [[a,a,a,...,a],[t,t,t,...,t]]
        for instruction, [action, target] in episode:
            normalized = preprocess_string(instruction).split()
            #print(len(normalized))
            for word in normalized:
                if word in v2i:
                    ins.append(v2i[word])
                else:
                    ins.append(3)
            actionl.append(a2i[action])
            targetl.append(t2i[target])

        # append end token
        ins.append(2)
        actionl.append(2)
        targetl.append(2)

        # pad the instruction sequence
        while (len(ins)<ins_pad_length):
            ins.append(0)
        while (len(targetl)<target_pad_length):
            targetl.append(0)
        while (len(actionl)<action_pad_length):
            actionl.append(0)

        #if last_a_length != len(actionl):
        #    print("different action sequence length!", len(actionl))
        #if last_t_length != len(targetl):
        #    print("different target sequence!", len(targetl))
        #if len(actionl)>maxa:
        #    maxa = len(actionl)
        #if len(targetl)>maxt:
        #    maxt = len(targetl)
        
        #last_a_length = len(actionl)
        #last_t_length = len(targetl)
        
        #print("max", maxa,maxt)
        #append to train_list
        insi = torch.IntTensor(ins)
        targeti = torch.IntTensor(targetl)
        actioni = torch.IntTensor(actionl)
        target_tensor = torch.IntTensor([actionl,targetl])
        insi.to(device)
        targeti.to(device)
        actioni.to(device)
        target_tensor.to(device)

        train_list.append((insi,target_tensor))

    # same thing for validation data
    for episode in val_all_string:
       # each entry in an episode is processed individually
        ins = []
        actionl = []
        targetl = []

        # append start token
        ins.append(1)
        actionl.append(1)
        targetl.append(1)

        # since doing seq2seq, need to combine all single entries in episode into a single sequence
        # in the form of ["compounded_low_level instructions", [[a,a,a,...,a],[t,t,t,...,t]]
        for instruction, [action, target] in episode:
            normalized = preprocess_string(instruction).split()
            #print(len(normalized))
            for word in normalized:
                if word in v2i:
                    ins.append(v2i[word])
                else:
                    ins.append(3)
            actionl.append(a2i[action])
            targetl.append(t2i[target])

        # append end token
        ins.append(2)
        actionl.append(2)
        targetl.append(2)

        # pad the instruction sequence
        while (len(ins)<ins_pad_length):
            ins.append(0)
        while (len(targetl)<target_pad_length):
            targetl.append(0)
        while (len(actionl)<action_pad_length):
            actionl.append(0)
        

        #append to train_list
        insi = torch.IntTensor(ins)
        targeti = torch.IntTensor(targetl)
        actioni = torch.IntTensor(actionl)
        target_tensor = torch.IntTensor([actionl,targetl])
        insi.to(device)
        targeti.to(device)
        actioni.to(device)
        target_tensor.to(device)


        val_list.append((insi,target_tensor))
    
    #for insi,target_tensor in val_list:
    #    print("val: ",target_tensor[0],target_tensor[1])
    #print("val: ",val_list[0][1][0],val_list[0][1][1]) 

    print("train list size: ", len(train_list))
    print("val list size: ",len(val_list))

    # to save my life
    train_list = train_list[:7000]
    val_list = val_list[:1500]


    train_loader = DataLoader(train_list, batch_size=args.batch_size, shuffle=True)

    val_loader = DataLoader(val_list, batch_size=args.batch_size, shuffle=True)

    return train_loader, val_loader, (v2i, i2v, a2i, i2a, t2i, i2t)


def setup_model(args, map, device):
    """
    return:
        - model: YourOwnModelClass
    """
    # ================== TODO: CODE HERE ================== #
    # Task: Initialize your model. Your model should be an
    # an encoder-decoder architecture that encoders the
    # input sentence into a context vector. The decoder should
    # take as input this context vector and autoregressively
    # decode the target sentence. You can define a max length
    # parameter to stop decoding after a certain length.

    # For some additional guidance, you can separate your model
    # into an encoder class and a decoder class.
    # The encoder class forward pass will simply run the input
    # sequence through some recurrent model.
    # The decoder class you will need to implement a teacher
    # forcing mechanism in the forward pass such that instead
    # of feeding the model prediction into the recurrent model,
    # you will give the embedding of the target token.
    # ===================================================== #
    #print(map)
    (v2i, i2v, a2i, i2a, t2i, i2t) = map
    #print("action dict: ",a2i,"target dict: ",t2i)
    input_dim = len(v2i)
    output_dim = len(a2i)+len(t2i)
    target_size = [len(a2i),len(t2i)]
    embedding_dim = 256
    hidden_dim = 512
    src_vocab_size = len(v2i)
    trg_vocab_size=[len(a2i),len(t2i)]
    src_pad_idx = 0
    trg_pad_idx = 0

    #print(decoder)

    model = Transformer(
        src_vocab_size,trg_vocab_size,src_pad_idx,trg_pad_idx
    ).to(device)
    return model


def setup_optimizer(args, model):
    """
    return:
        - criterion: loss_fn
        - optimizer: torch.optimvi
    """
    # ================== TODO: CODE HERE ================== #
    # Task: Initialize the loss function for action predictions
    # and target predictions. Also initialize your optimizer.
    # ===================================================== #
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(),lr=0.00001)

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
    """
    # TODO: implement function for greedy decoding.
    # This function should input the instruction sentence
    # and autoregressively predict the target label by selecting
    # the token with the highest probability at each step.
    # Note this is slightly different from the forward pass of
    # your z because you want to pick the token
    # with the highest probability instead of using the
    # teacher-forced token.

    # e.g. Input: "Walk straight, turn left to the counter. Put the knife on the table."
    # Output: [(GoToLocation, diningtable), (PutObject, diningtable)]
    # Also write some code to compute the accuracy of your
    # predictions against the ground truth.
    """

    epoch_loss = 0.0
    epoch_acc = 0.0

    # iterate over each batch in the dataloader
    # NOTE: you may have additional outputs from the loader __getitem__, you can modify this
    for (inputs, labels) in loader:
        # put model inputs to device
        inputs, labels = inputs.to(device), labels.to(device)

        # calculate the loss and train accuracy and perform backprop
        # NOTE: feel free to change the parameters to the model forward pass here + outputs
        aoutput,toutput = model(inputs, labels)
        #print("training forward pass",labels[:,0].shape,labels[:,0],labels[:,1].shape,labels[:,1],aoutput.squeeze().shape,toutput.squeeze().shape)

        aoutput = torch.transpose(aoutput,1,2)
        toutput = torch.transpose(toutput,1,2)

        aloss = criterion(aoutput.squeeze(), labels[:,0].long())
        tloss = criterion(toutput.squeeze(), labels[:,1].long())

        loss=aloss+tloss

        #print(loss.shape,loss)

        # step optimizer and compute gradients during training
        if training:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        """
        # TODO: implement code to compute some other metrics between your predicted sequence
        # of (action, target) labels vs the ground truth sequence. We already provide 
        # exact match and prefix exact match. You can also try to compute longest common subsequence.
        # Feel free to change the input to these functions.
        """
        # TODO: add code to log these metrics
        #em = [aoutput,toutput] == labels
        #prefix_em = prefix_match([aoutput,toutput], labels)
        aacc = 0.0
        tacc = 0.0
        
        #print(aoutput.shape,toutput.shape,labels.shape,labels[:,0].shape,labels[:,1].shape)

        aem = torch.transpose(aoutput,1,2).argmax(-1)
        alen = len(aem)
        for i in range(len(aem)):
            aprefix = custom_match(aem[i],labels[:,0][i])
            aacc += aprefix
        tem = torch.transpose(toutput,1,2).argmax(-1)
        tlen = len(tem)
        for i in range(len(tem)):
            tprefix = custom_match(tem[i],labels[:,1][i])
            tacc += tprefix
        #aprefix = prefix_match(aoutput,labels[:,0])
        #tprefix = prefix_match(toutput,labels[:,1])
        #print("prior to adding: ",aacc,tacc,len(em))
        #print(aacc,tacc,alen,tlen)
        acc = aacc/alen+tacc/tlen

        # logging
        epoch_loss += loss.item()
        #epoch_acc += acc.item()
        epoch_acc += acc

    epoch_loss /= len(loader)
    epoch_acc /= len(loader)

    return epoch_loss, epoch_acc


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


def train(args, model, loaders, optimizer, criterion, device):
    # Train model for a fixed number of epochs
    # In each epoch we compute loss on each sample in our dataset and update the model
    # weights via backpropagation

    # Store Information for graph
    tl = []
    vl = []
    va = []

    model.train()
    for epoch in tqdm(range(int(args.num_epochs))):

        # adding line to set model to train after eval set it to .eval()
        model.train()

        # train single epoch
        # returns loss for action and target prediction and accuracy
        train_loss, train_acc = train_epoch(
            args,
            model,
            loaders["train"],
            optimizer,
            criterion,
            device,
        )

        # some logging
        print(f"train loss : {train_loss}")
        tl.append(train_loss)

        # run validation every so often
        # during eval, we run a forward pass through the model and compute
        # loss and accuracy but we don't update the model weights
        if epoch % int(args.val_every) == 0:
            val_loss, val_acc = validate(
                args,
                model,
                loaders["val"],
                optimizer,
                criterion,
                device,
            )

            print(f"val loss : {val_loss} | val acc: {val_acc}")
            vl.append(val_loss)
            va.append(val_acc)

    # ================== TODO: CODE HERE ================== #
    # Task: Implement some code to keep track of the model training and
    # evaluation loss. Use the matplotlib library to plot
    # 3 figures for 1) training loss, 2) validation loss, 3) validation accuracy
    # ===================================================== #
    trainepoch = range(0,len(tl))
    valepoch = range(0,len(vl))

    plt.plot(trainepoch, tl)
    plt.xlabel("training epoch")
    plt.ylabel("training loss")
    plt.title("training loss")
    plt.savefig("./statistic-transformer/training.pdf")
    plt.clf()

    plt.plot(valepoch, vl, label="val loss")
    plt.plot(valepoch, va, label="val acc")
    plt.title("validation")
    plt.savefig("./statistic-transformer/val.pdf")
    plt.clf()



def main(args):
    device = get_device(args.force_cpu)
    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # get dataloaders
    train_loader, val_loader, maps = setup_dataloader(args)
    loaders = {"train": train_loader, "val": val_loader}

    # build model
    model = setup_model(args, maps, device)
    print(model)

    # get optimizer and loss functions
    criterion, optimizer = setup_optimizer(args, model)

    if args.eval:
        val_loss, val_acc = validate(
            args,
            model,
            loaders["val"],
            optimizer,
            criterion,
            device,
        )
    else:
        train(args, model, loaders, optimizer, criterion, device)


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
    parser.add_argument("--num_epochs", default=1000, help="number of training epochs")
    parser.add_argument(
        "--val_every", default=5, help="number of epochs between every eval loop"
    )

    # ================== TODO: CODE HERE ================== #
    # Task (optional): Add any additional command line
    # parameters you may need here
    # ===================================================== #
    args = parser.parse_args()

    main(args)
