import tensorflow as tf
import numpy as np
from keras.optimizers import RMSprop
from src.utils import getJsonDataFromConfigFile,one_hot
from keras.layers import Embedding, LSTM, Dense, Dropout
from keras.callbacks import EarlyStopping, Callback
from keras.models import Sequential

class RNN:
    def __init__(self, vocab_size, model):
        self.model=model
        #self.embeddings = embeddings
        self.config = getJsonDataFromConfigFile("config.json")
        self.learning_rate = self.config["RNN"]["learning_rate"]
        # self.batch_size=self.config["alg"]["batch_size"]
        self.batch_size = self.config["alg"]["batch_size"]
        self.is_embedding = self.config["RNN"]["embedding"]["used"]
        # length of the input per time-step, for example input at time-step t has input [1,2] of length 2
        self.input_length = self.config["RNN"]["input_length"]
        self.ngram_length = self.config["ngram"]["length"]
        # number of units in RNN cell
        self.n_hidden1 = self.config["RNN"]["layers"]["layer0"]
        self.n_hidden2 = self.config["RNN"]["layers"]["layer1"]
        self.vocab_size = vocab_size
        # minus 1 because of ngram
        self.keep_prob = self.config["RNN"]["dropout"]
        self.x_input = self.x = tf.placeholder(tf.float32, [self.batch_size, self.ngram_length - 1], "x_ngram")
        self.y_input = self.y = tf.placeholder(tf.int64, [self.batch_size, self.vocab_size], name="y_ngram")
        # RNN output node weights and biases


    def __call__(self, *args, **kwargs):
        if self.model is None:
            self.model = Sequential()
            # return sequence in the first LSTM layer to pass the sequences to the second layer
            self.model.add(LSTM(self.n_hidden1, return_sequences=True,input_shape=(self.ngram_length-1, self.vocab_size),stateful=False))
            # using dropout to avoid overfitting
            self.model.add(Dropout(self.keep_prob))
            self.model.add(
                LSTM(self.n_hidden2, input_shape=(self.ngram_length - 1, self.vocab_size),
                     stateful=False))
            self.model.add(Dropout(self.keep_prob))

            self.model.add(Dense(self.vocab_size, activation='softmax'))

            optimizer = RMSprop(lr=self.learning_rate)
            self.model.compile(loss='categorical_crossentropy', optimizer=optimizer)


    def get_accuracy(self, batch_data, batch_labels, sess):
        # train the model with the ngrams and their predictors and labels
        earlystop = EarlyStopping(monitor='val_loss', min_delta=0, patience=5, verbose=0, mode='auto')

        self.model.fit(batch_data, batch_labels,
                  batch_size=self.batch_size,
                  epochs=1,callbacks=[EvalCallback((batch_data, batch_labels),self.model)])
        return self.model


    def predict(self, batch_data, batch_labels):
        # use saved model if predict() called without train() before
        batch=np.reshape(batch_data,(1,self.ngram_length-1,self.vocab_size))
        preds = self.model.predict(batch, verbose=2)
        return preds


class EvalCallback(Callback):
    def __init__(self, test_data,model):
        self.test_data = test_data
        self.model=model

    def on_epoch_end(self, epoch, logs={}):
        batch_data, batch_labels = self.test_data
        preds = self.model.predict(batch_data, verbose=2)
        print("preds.shape=", np.shape(preds))
        print("argmax(preds,1)=")

        equal = np.equal(np.argmax(preds, 1), np.argmax(batch_labels, 1))
        d = [i for i in equal if i == True]
        accuracy = len(d) / batch_labels.shape[0]
        print("accuracy=", accuracy)
        for i, j in zip(np.argmax(preds, 1)[:30], np.argmax(batch_labels, 1)[:30]):
            print("predicted=", i, " - label=", j)
        print("accuracy=", accuracy)

class DNN:
    def __init__(self, n_input_cbows, max_hole_size):
        self.config = getJsonDataFromConfigFile("config.json")
        self.learning_rate = self.config["DNN"]["learning_rate"]
        self.n_input_cbows = n_input_cbows
        self.max_hole_size = max_hole_size
        self.x_cbows = tf.placeholder("float", [None, n_input_cbows * 2], name="x_cbows")
        self.y_cbows = tf.placeholder("float", [None, max_hole_size], name="y_cbows")
        self.keep_prob = tf.placeholder("float", name="dropout")
        self.dnn_variables=[]

    def __call__(self, *args, **kwargs):
        with tf.variable_scope("dnn_scope"):
            # reshape to [1, n_input]
            # self.x = tf.reshape(self.x, [-1, self.n_input_cbows*2])
            self.x_cbows = tf.reshape(self.x_cbows, [-1, self.n_input_cbows * 2])


            hidden = tf.layers.dense(inputs=self.x_cbows, units=self.config["DNN"]["layers"]["layer0"],
                                     activation=tf.nn.relu)
            # to avoid overfitting dropout is used
            hidden = tf.nn.dropout(hidden, keep_prob=self.keep_prob)
            hidden = tf.layers.dense(inputs=hidden, units=self.config["DNN"]["layers"]["layer1"],
                                     activation=tf.nn.relu)
            # to avoid overfitting dropout is used
            hidden = tf.nn.dropout(hidden, keep_prob=self.keep_prob)
            # 3 classes => 3 hole sizes to predict
            self.out = tf.layers.dense(inputs=hidden, units=self.max_hole_size, activation=tf.nn.softmax)

            # Loss and optimizer2
            self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.out, labels=self.y_cbows))
            self.optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(self.cost)

            # Model evaluation
            self.correct_pred = tf.equal(tf.argmax(self.out, 1), tf.argmax(self.y_cbows, 1))
            self.accuracy = tf.reduce_mean(tf.cast(self.correct_pred, tf.float32))

            # there are n_input outputs but
            # we only want the last output
            return self.out


    def predict_hole_size(self, batch_data, sess, keep_prob):
        pred = sess.run(self.out, feed_dict={self.x_cbows: batch_data, self.keep_prob: keep_prob})
        return pred

    def optimize(self, batch_data, batch_labels, sess, keep_prob):
        # Loss and optimizer
        _, cost_eval = sess.run([self.optimizer, self.cost],
                                feed_dict={self.x_cbows: batch_data, self.y_cbows: batch_labels,
                                           self.keep_prob: keep_prob})
        return cost_eval

    def get_accuracy(self, batch_data, batch_labels, sess, keep_prob):
        # Model evaluation
        onehot_pred, pred_eval, accuracy = sess.run([self.out, self.correct_pred, self.accuracy],
                                                    feed_dict={self.x_cbows: batch_data, self.y_cbows: batch_labels,
                                                               self.keep_prob: keep_prob})
        # onehot_pred = [one_hot([int(tf.argmax(i).eval())], np.arange(3)) for i in onehot_pred]
        return pred_eval, onehot_pred, accuracy
