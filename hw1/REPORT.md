For the model, its implementation I still choose to go with two prediction heads of an LSTM model.
The reason for that is one that makes more sense to me, as the two targets are based on the same set of inputs,
with only difference in label, which can be adjusted and adpated to, no need to create two models for this.

However for the parameter choices, they are sort of random. I don't have the time to play around with them
just yet due to all sorts of things. But 128 seems like a good number for hidden dimension, and from there
I went on to adjust other hyperparameteres so everything fits with the batchsize and all dimensions have no problem
while running the program.

For loss impelentation, I looked into it, and after some searching I came to the conclusion that loss more than 1
with log_softmax and NLLLoss is not necessarily a problematic thing, but NLLLoss is not normalized, that is why.

However performance wise there is something need to be said. Training (for testing resons) is limited to a small dataset
of 5000 of the original set, which means that it has a much smaller set to train from. It resulted in a rather good and fast
training for action set as there are way fewer actions than targets provided in the original dataset. For some reason,
target training is way off. I suspect it might be due to the difference in range, layer sizes needs to change as well
for target. However they are the same right now.

Then moving on to validation set. With a small testing set, it can be deduced that validation doesn't work as good as training,
but the general trend was kept between training and validation: action have a higher acc and training speed, but for target
training it is slow and the rate doesn't seem like it is learning all that fast. I would attribute it to the conclusion I 
came to in the last paragraph, however, that needs to be tested.

Final figures would be produced as a pdf.