Something to note (note to myself, some problem found during implementation): 

1. encoding enforcing, somehow my pc uses gbk if not enforced (probaboly due to having 
simplified chinese as one of the language, utf-8 enforcing is favored. Took a while to figure it out)

2. current pytorch version installed by requirement.txt seems to not be able to run on a Nvidia A100
with sm80, the newest version which supports cuda 11.6 solves it.

3. also model while running on cuda prompted an error saying not all tensors are in the same place,
which in my code there is a part where model is being forced into cuda to solve this problem. But
maybe it is also part of the cuda problem? Did not test them separately


The model implemented here is a CBOW model with the embedding layer going from 3000 (vocab size) to
128, then the embeddings are summed (by using numpy.mean on sums of contexts so they are summed then
mapped to embedding_dimension) and fed through a linear layer where then ReLU just in an attempt to try
to better performance, and the output is fed into a final linear layer which size is 128 - 3000, serving
as the final output layer.

The loss function in choise is cross entropy loss function, and due to the way that cross entropy loss
is implemented in pytorch, no softmax or NLL was applied individually, instead just fed the linear layer's
output straight to the loss function for calculation.

For preparing the data for training, due to the nature of CBOW, a context-to-target_word format of training
data is used. Take the books given as raw data, parse them into individual lines and stored them as a vector
of vectors. Then they are encoded using the function given in utils.py. After which, every single line is parsed
through once to first get rid of the extra padding at the end, then to create the context and target_word 
pairings for training and validation. To do that, first pad the sentense based on the context window size,
e.g. if the context window size is 4, then put 2 pad tokens before the start and after the end of the sentence,
so while they are learning every single meaningful token can be learned. Then based on the context window, 
context vector is created, and the middle word is singled out, so this pairing can be stored as a 
tensor - int vector, and these are then stored in a vector of vectors. Random shuffle it then split it to train
and valid set based on a 70-30 ratio.

As for evaluation, for in vitro tasks we are testing for word prediciton, whether or not, for the CBOW model,
does it predict the target word given the context around it. For in vivo task it is given two words, predict 
whether or not the two words are analogies? (the abcd mentioned in downstream evaluation) And the words are
chosen for different categories for either sem or syn relation. I am not entirly sure on how these works (as
you can see there is fair amount of guessing in this paragraph), but the local extrema made me think whether
the in vitro evaluation at its current stage is best for evaluating the result. However that is still a guess
though since I cannot figure out why is there a local extrema (more on that in the next paragraph), and with
in vitro evaluation looking at guessing accuracy over vocabulary per batch, there shouldn't be much of a problem.

I've tested out different context windows sizes to see if they differ in results. Well during the training
there were difference in loss and accuracy, but that is almost miniscule with the difference ranging
within +/- 2% from what I see. The interesting thing is that with window context size 2 and 4, the acc/loss
curve exist a local max/min that the model achieved before the end of the training, and with window size 2
the extreme point appear to be earlier than window context size 4. Not sure what happend here, as I expected
a higher accuracy and lower loss with bigger window sizes. The guess here would be that due to the extended
window size, more tokens included padding which might have interfered with the training, or due to the longer
context, it may not be able to predict things as well as a suitable one for more complex information. Maybe
it will converge at a higher acc and lower loss if the epoch increases, but not sure. However for downstream
evaluation, it did show the trend that with a bigger window the result is better. Possible due to amount of
information that it sees in the part of sentence, therefore able to better assume the relation based on context?

Window Size 2 Result
...Total performance across all 1309 analogies: 0.0138 (Exact); 0.0295 (MRR); 34 (MR)
...Analogy performance across 969 "sem" relation types: 0.0114 (Exact); 0.0241 (MRR); 42 (MR)
        relation        N       exact   MRR     MR
        capitals        1       1.0000  1.0000  1
        binary_gender   12      0.5000  0.5448  2
        antonym 54      0.0741  0.0983  10
        member  4       0.0000  0.0056  177
        hypernomy       542     0.0000  0.0125  80
        similar 117     0.0000  0.0121  83
        partof  29      0.0000  0.0270  37
        instanceof      9       0.0000  0.0019  525
        derivedfrom     133     0.0000  0.0080  124
        hascontext      32      0.0000  0.0019  515
        relatedto       10      0.0000  0.0024  416
        attributeof     11      0.0000  0.0176  57
        causes  6       0.0000  0.0053  189
        entails 9       0.0000  0.0056  180
...Analogy performance across 340 "syn" relation types: 0.0206 (Exact); 0.0450 (MRR); 22 (MR)
        relation        N       exact   MRR     MR
        adj_adv 22      0.0000  0.0023  436
        comparative     7       0.0000  0.0344  29
        superlative     3       0.0000  0.0258  39
        present_participle      62      0.0484  0.0639  16
        denonym 2       0.0000  0.0445  22
        past_tense      64      0.0469  0.0873  11
        plural_nouns    107     0.0000  0.0252  40
        plural_verbs    73      0.0137  0.0357  28

Window Size 4 Result
...Total performance across all 1309 analogies: 0.0191 (Exact); 0.0399 (MRR); 25 (MR)
...Analogy performance across 969 "sem" relation types: 0.0083 (Exact); 0.0222 (MRR); 45 (MR)
        relation        N       exact   MRR     MR
        capitals        1       0.0000  0.3333  3
        binary_gender   12      0.1667  0.3223  3
        antonym 54      0.0370  0.0595  17
        member  4       0.0000  0.0094  107
        hypernomy       542     0.0037  0.0137  73
        similar 117     0.0171  0.0299  33
        partof  29      0.0000  0.0170  59
        instanceof      9       0.0000  0.0571  18
        derivedfrom     133     0.0000  0.0121  83
        hascontext      32      0.0000  0.0025  397
        relatedto       10      0.0000  0.0010  1009
        attributeof     11      0.0000  0.0193  52
        causes  6       0.0000  0.0223  45
        entails 9       0.0000  0.0099  101
...Analogy performance across 340 "syn" relation types: 0.0500 (Exact); 0.0902 (MRR); 11 (MR)
        relation        N       exact   MRR     MR
        adj_adv 22      0.0000  0.0107  93
        comparative     7       0.1429  0.1673  6
        superlative     3       0.0000  0.0156  64
        present_participle      62      0.0484  0.0915  11
        denonym 2       0.0000  0.0130  77
        past_tense      64      0.0625  0.1333  8
        plural_nouns    107     0.0654  0.0919  11
        plural_verbs    73      0.0274  0.0706  14

Window Size 6 Result
...Total performance across all 1309 analogies: 0.0260 (Exact); 0.0490 (MRR); 20 (MR)
...Analogy performance across 969 "sem" relation types: 0.0093 (Exact); 0.0233 (MRR); 43 (MR)
        relation        N       exact   MRR     MR
        capitals        1       0.0000  0.2000  5
        binary_gender   12      0.2500  0.3600  3
        antonym 54      0.0185  0.0568  18
        member  4       0.0000  0.0099  101
        hypernomy       542     0.0055  0.0148  68
        similar 117     0.0085  0.0275  36
        partof  29      0.0000  0.0283  35
        instanceof      9       0.0000  0.0069  144
        derivedfrom     133     0.0000  0.0091  109
        hascontext      32      0.0000  0.0051  198
        relatedto       10      0.0000  0.0085  117
        attributeof     11      0.0000  0.0035  282
        causes  6       0.0000  0.0464  22
        entails 9       0.1111  0.1161  9
...Analogy performance across 340 "syn" relation types: 0.0735 (Exact); 0.1221 (MRR); 8 (MR)
        relation        N       exact   MRR     MR
        adj_adv 22      0.0000  0.0091  110
        comparative     7       0.0000  0.1780  6
        superlative     3       0.0000  0.0542  18
        present_participle      62      0.1129  0.1487  7
        denonym 2       0.0000  0.0298  34
        past_tense      64      0.0625  0.1387  7
        plural_nouns    107     0.0748  0.1214  8
        plural_verbs    73      0.0822  0.1202  8

Corrected CBOW with context window 6
...Total performance across all 1309 analogies: 0.0558 (Exact); 0.0911 (MRR); 11 (MR)
...Analogy performance across 969 "sem" relation types: 0.0124 (Exact); 0.0314 (MRR); 32 (MR)
        relation        N       exact   MRR     MR
        capitals        1       0.0000  0.5000  2
        binary_gender   12      0.1667  0.3814  3
        antonym 54      0.0370  0.0615  16
        member  4       0.0000  0.0071  141
        hypernomy       542     0.0074  0.0206  49
        similar 117     0.0171  0.0332  30
        partof  29      0.0000  0.0232  43
        instanceof      9       0.1111  0.1146  9
        derivedfrom     133     0.0075  0.0258  39
        hascontext      32      0.0000  0.0072  138
        relatedto       10      0.0000  0.0512  20
        attributeof     11      0.0000  0.0331  30
        causes  6       0.0000  0.0688  15
        entails 9       0.0000  0.0330  30
...Analogy performance across 340 "syn" relation types: 0.1794 (Exact); 0.2612 (MRR); 4 (MR)
        relation        N       exact   MRR     MR
        adj_adv 22      0.0000  0.0500  20
        comparative     7       0.5714  0.6203  2
        superlative     3       0.0000  0.2133  5
        present_participle      62      0.1935  0.2635  4
        denonym 2       0.0000  0.2507  4
        past_tense      64      0.2969  0.3783  3
        plural_nouns    107     0.1682  0.2498  4
        plural_verbs    73      0.1096  0.2049  5