# Software-Analysis-with-Deep-Learning

## Installation
The completion process can be run with the trained model without modification in the code by executing runner.py from the terminal and from the directory CourseProject. 
```
python3 -m src.runner
```
To train the model and overwrite the old model files, the variable use_stored_model in runner.py has to be set to False first.

## Requirements
```
keras
tensorflow
```
## 1 Approach
The algorithm of this implementation runs over several phases with different methods. It first predicts the hole size and then predicts the missing tokens of the hole.
### 1.1 Hole size prediction
The methods used for this task is CBOW and a Feed Forward Neural Network as classifier. As the maximum number of expected tokens in the hole is three, the Neural Network has three classes where each class equals a hole size. The Neural Network gets k Tokens as CBOW and as a number. The CBOW-samples are generated from the training data. Mini batch are used and shuffled to avoid overfitting.
### 1.2 Prediction of missing tokens
The missing tokens are predicted by two separately trained LSTM-RNN, which is an appropriate choice for this task. As input the Neural Networks gets tokens as ngram. One Neural Network is used for predicting the suffix and the other is trained with reverted ngram.
The Network Models for forward and backward prediction are built the same way. It is comprised of LSTM layers with alternating dropout layer and finally with a dense layer for constraining the number of outputs to the number of unique tokens. At the end there is a softmax layer for predicting the probabilities of the next token.
For predicting the tokens in the hole the same token of the forward and backward prediction are multiplied and the token with the highest common probability is taken.
## 2 Implementation
The RNN is implemented with the tensorflow-wrapper Keras and the DNN for predicting the hole size is implemented with pure tensorflow.
Both network have their CBOW and ngram input padded with zeros whenever necessary. Saving and loading of the model is supported.
A json config file is used for changing the parameters of the networks.
## 3 Evaluation
### 3.1 Parameters
The three Neuronal Networks are run with different parameters for evaluation purpose. The parameters tried are layer size, number of neurons in every layer, dropout ratio, learning rate and the size of the ngram and CBOW.
Increasing number of LSTM-Layer and dropout didn’t have a significant impact on the performance, but to generally avoid overfitting dropout, they are still used in the final experiment. Both Networks, DNN and RNN use two hidden layers and dropout.
### 3.2 Performance
The RNN next token prediction has an accuracy of over 60 % and the DNN hole size prediction has an average accuracy of 50 %. The overall accuracy is about 25%.
The training is run only over one epoch due to high usage of processor resources. More epochs could lead to higher performance
## 4 Possible further improvements
Methods which could further improve the performance is the additional usage of ngram probability estimation and to take the joint prediction of it and RNN, like adapted in the recommended paper for this course. A bad performance for ngram estimation would have bad impact on the total performance. The same is true for using CBOW probability estimation additionally to the Feed Forward Neural Network prediction of the hole size. Word Embeddings could also lead to a performance increase.
An appropriate Word Embedding could improve the performance. There is a Python Word Embedding available, but no Javascript, which could still lead to higher performance


