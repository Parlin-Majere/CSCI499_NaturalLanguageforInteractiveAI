Something to note

1. Again, there is a problem with training the model on GPU. Apparently the initial setup does not allow every tensor
to be loaded onto GPU, some manipulation is needed at least for me to make the thing run on GPU. Maybe this will be something
to look for in the future when building the initial framework for the model.


Introduction to assignment structures

The hw3 folder here contains 2 training code (train.py, transformertrain.py), three models (model.py, attnmodel.py, transformermodel.py)
corresponding to Encoder-Decoder LSTM model, Encoder-Attention Decoder LSTM model, and a transformer model. Then, there are a few folders
containing the training tracking pdf's, which you can defer from name what model they correspond to.

For the encoder-decoder LSTM model, instead of making encoder and decoder just the LSTM/RNN models they are as some structure explained,
I still chose to put everything they need (hidden layer, embedding layer, fc layers and etc) in there, not because doing this has an
advantage over other methods, it is because it preserves the structure from hw1. Same goes for the one using attention decoder. However
transformer model is built from "scratch". I didn't get the time to get familiar with huggingface so I did not pull a model that is already 
built from them, instead I referenced Aladdin Persson's implementation of a transformer based encoder-decoder model. Then alterations were
made so that it fits our purpose of one seq -> two seq's prediction. But that baseline model is quite similar to the structure introduced 
in lecture/on slides.

For data preprocessing, I choose to do it as 3 sequences. For each episode which contains number of instruction -> (action, target)
pairings, which are each put together into a sequence corresponding to each category of input. Then the instruction sequence is used
for input as the original language in machine translation, and then the model would translate the sequence into two different "languages":
action sequence and target sequence. There is a changed made to utils.py here, which added start, end and padding token to action/target
sequence just like the instruction sequences, otherwise it can't really be treated as a sequence for seq2seq models.

As for hyperparameters there ain't really a reason of why I chose them, the embedding size, hidden dimension and etc are all inherited from
past assignments where they showed fine results. Batch size used is 64 since running on a better cpu and it would fit, validating every epoch
due to an ealier error that caused my model to overfit and I need to check every single one just to make sure what happened. Tailored training
and validation data to save some time for training, since if training the whole dataset with 100 epoch, the model will be running for more than
4 hrs even on a good GPU.

For the encoder-decoder LSTM models, one thing needs to be noted is that since the decoder has 2 prediction heades, it has two distinct fc layers
for outputing predictions. Embeddings are shared in a way of concatenation, where target and action has their own embedding layers, but then
they are concatenated and fed into the LSTM. From this point on everything in the decoder is shared until the final output layers. Dropout layers
are added to avoid overfitting, which as said is a problem that poped up during early development phase of this model. You will be seeing some 
transpositions of tensors with in both models, but that is a problem due to, well, construction of dataloader, input and output parametere of
different layers within the model. They do not affect performance, but it is just something to keep an eye on.

The default attention (first implemented) is a global attention for encoder - attention decoder, and there is a local attention implementation
contained as well, both being soft as I couldn't figure out how to make the hard attention work within the next 40 minutes that I have left when I 
write this line down. Upon some research I found that hard attention with monte-carlo sampling method would be a possibility to make the hard
attention differentiable for backpropagation at the end of each episode. However hard attention is normally performing worse than soft attention,
thus not used that often. (Jonathan Hui, https://jhui.github.io/2017/03/15/Soft-and-hard-attention/) Then there is another method proposed by
Shiv Shankar, Siddhant Garg and Sunita Sarawagi in paper "Surprisingly Easy Hard-Attention for Sequence to Sequence Learning" which provides
an alternative to hard-attention with some sampling method. The thing is even I would like to implement the said methods, I do not have the time
to write my code or to test the model. But both of these methods are suggesting to produce a softmax or some other differentiable layer by sampling
the whole episode after applying hard-attention, therefore the differentiable layer is still representable of the training, and that also allows
it to backpropagate as normal. But for comparison between global-soft and local-soft, it is quite easy to see that local-soft has a better performance. 
Local attention is implemented through calculation of window location (start and end index) within encoder's output for each specific token in the output 
sequence, and it is not even a complete mapping, as a a few tokens within the encoder output sequence is being ignored by this calculation method.
However I would say that this does not cause much performance, as upon checking the dataloader sequences, the end is almost certainly going to be 
a bunch of padding tokens, which are there because I did not want to cut off any token from any sequences, so used a long enough max_length to make 
sure everything is included. And it is worth noting that both attention performs better than simple LSTM models, given that yes, it would take more 
time to train.

During training, teacher-forcing is only applied 50% of the time with implementation of a random function that controls when teacher forcing would
be applied. Doing this because otherwise, if we are always doing teacher forcing, overfitting is going to be over the roof. The model will
never be able to test itself on validation dataset, and the val_loss just kept on growing as train_loss consistently drops. So that is a necessary
feature to be implemented.

Then for val_acc, it is like loss where action and target are calculated differently and then added together to represent the whole episode,
however I've changed the matching function to be one that only accounts for the tokens that are meaningful, which means that the function
only considers tokens that are non-padding, including "sos" and "eos" tokens, and it stops after comparing eos of the target sequence to 
one that predicted by the model.

Moving on to transformers. There aren't much to talk about here, as it is a pretty basic transformer based encoder decoder model. The only thing
that differs it from the standard version of such model is that, like the LSTM models, it included two prediction heads in decoder. And much are 
the same as well. However just for safety reasons, I made them so that they are also independent of each other during decoder blocks, and this time
they are not sharing an embedding layer, meaning that there is no concatenation going on here. Also a dropout of 0.2 is implemented, just as
another cautionary step against overfitting.

For performance of transformer model, though it is not logged, the running log is posted below. As we can see it started off poorly compare to
LSTM based models, however it learns fast, reaching 0.5 acc in 10 epochs. And from my past experience working with ViT's, when compared to CNN,
they usually outperforms CNN (given that the ViT is well written). And if not in early epochs, transformer based models have apparently a higher
cap, while traditional methods start to show sign of convergence, transformers are still doing well. I suppose it is what happened with NLP
tasks as well.


**Transformer training and validation result, which failed to log due to I forgot to include matplotlib
  0%|                                                                                                                                                                                                                                        | 0/10 [00:00<?, ?it/s]train loss : 7.7729507706382055
val loss : 7.239373981952667 | val acc: 0.033002614417768375
 10%|██████████████████████▎                                                                                                                                                                                                        | 1/10 [04:02<36:26, 242.94s/it]train loss : 7.0195212494243275
val loss : 6.51091988881429 | val acc: 0.03412798644157789
 20%|████████████████████████████████████████████▌                                                                                                                                                                                  | 2/10 [08:04<32:15, 241.90s/it]train loss : 6.293233420632102
val loss : 5.815743386745453 | val acc: 0.03777182679021735
 30%|██████████████████████████████████████████████████████████████████▉                                                                                                                                                            | 3/10 [12:05<28:13, 241.89s/it]train loss : 5.60295560576699
val loss : 5.156406144301097 | val acc: 0.14251770642879824
 40%|█████████████████████████████████████████████████████████████████████████████████████████▏                                                                                                                                     | 4/10 [16:07<24:11, 241.90s/it]train loss : 4.952354231747714
val loss : 4.5450116991996765 | val acc: 0.16218160912656873
 50%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████▌                                                                                                               | 5/10 [20:10<20:11, 242.24s/it]train loss : 4.361015419526534
val loss : 3.988157441218694 | val acc: 0.21483182071725107
 60%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▊                                                                                         | 6/10 [24:11<16:07, 241.84s/it]train loss : 3.8203324101187968
val loss : 3.492328941822052 | val acc: 0.3579644691744219
 70%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████                                                                   | 7/10 [28:19<12:10, 243.61s/it]train loss : 3.3468367424878207
val loss : 3.060409724712372 | val acc: 0.5195867298560229
 80%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▍                                            | 8/10 [32:24<08:08, 244.35s/it]train loss : 2.9315611362457275
val loss : 2.68375962972641 | val acc: 0.7266749851862607
 90%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▋                      | 9/10 [36:29<04:04, 244.52s/it]train loss : 2.5789772965691307
val loss : 2.36942125360171 | val acc: 0.7581769523305026
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 10/10 [40:38<00:00, 243.83s/it]
Traceback (most recent call last):
  File "transformertrain.py", line 506, in <module>
    main(args)
  File "transformertrain.py", line 481, in main
    train(args, model, loaders, optimizer, criterion, device)
  File "transformertrain.py", line 441, in train
    plt.plot(trainepoch, tl)
NameError: name 'plt' is not defined