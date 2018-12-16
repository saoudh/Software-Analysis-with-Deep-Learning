import numpy as np
import collections
import os
import json
import tensorflow as tf
import sys
from keras.models import model_from_json
from keras.models import load_model

#settings_path_dir = "../settings/"
settings_path_dir = "./"



def getJsonDataFromConfigFile(file_name):
    # first argument-value is the setting-file-path
    if len(sys.argv) == 1:
        with open(settings_path_dir + file_name, 'r') as f:
            return json.load(f)
    else:
        with open(settings_path_dir + sys.argv[1], 'r') as f:
            return json.load(f)


def one_hot(idx_number, dictionary):
    if isinstance(idx_number,list):
        idx_number=idx_number[0]
    vector = [0] * len(dictionary)
    vector[idx_number] = 1
    return vector

def save_model_to_file_for_keras(model,model_file):
    jsn_model = model.to_json()

    with open(model_file+".json", 'w') as jsn:
        jsn.write(jsn_model)

    # serialize weights to HDF5
    model.save_weights(model_file+'.h5')

def load_model_from_file_for_keras(model_file):
    # open model file and if doesn't exists, then return None
    print("modelfile=",model_file)
    try:
        with open(model_file+'.json', 'r') as jsn_file:
            jsn_model = jsn_file.read()
    except FileNotFoundError:
        print("Filenotfound for ",model_file)
        return None


    model = model_from_json(jsn_model)

    # load weights into new model
    model.load_weights(model_file+'.h5')

    model.save(model_file+'.hdf5')
    model = load_model(model_file+'.hdf5')
    return model

def save_model_to_file(sess,model_file):
    # save model data only of the DNN by its scope to file
    var_list2 = [v for v in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="dnn_scope")]

    sess.run(tf.variables_initializer(var_list2))
    temp_saver = tf.train.Saver(
        var_list=[v for v in var_list2])

    temp_saver.save(sess, model_file,write_meta_graph=False, write_state=True)

def load_model_from_file(sess,model_file):
    # get scope of variables of the DNN by its scope
    var_list2 = [v for v in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="dnn_scope")]
    sess.run(tf.variables_initializer(var_list2))
    temp_saver = tf.train.Saver(
        var_list=[v for v in var_list2])
    # get directory path of the checkpoint file and set name of the file in the parameter latest_filename
    ckpt_state = tf.train.get_checkpoint_state(os.path.join("/".join(model_file.split("/")[:-1])), latest_filename="checkpoint")
    temp_saver.restore(sess, ckpt_state.model_checkpoint_path)
    return sess

def reversed_one_hot(onehot, dictionary):
    return dictionary[int(np.argmax(onehot))]


def token_to_string(token):
    return token["type"] + "-@@-" + token["value"]


def string_to_token(string):
    splitted = string.split("-@@-")
    return {"type": splitted[0], "value": splitted[1]}

def get_indexed_tokens_by_dict_data(begin_idx,end_idx,dict_data,reverse_dictionary):
    # get only last ngram of the prefix because we learned ngram prediction
    dict_data = dict_data[begin_idx:end_idx]
    # get tokens-dict-item as string of format "token-@-value"
    dict_data = [token_to_string(i) for i in dict_data]
    # get indexed integer value of the token-strings
    dict_data = [reverse_dictionary[i] for i in dict_data]

    return dict_data

class cbow_processing:
    def __init__(self,max_hole_size=3, n_input_cbows = 3):
        self.max_hole_size=max_hole_size
        self.n_input_cbows =n_input_cbows


    def prepare_cbows(self,data, dictionary):
        number_of_rounds = len(data) - self.n_input_cbows*2 -2

        X_f=[]
        Y_f=[]
        # hole_size 0 corresponds to the hole size of 1 and so on
        for hole_size in range(0,self.max_hole_size):
            X = np.zeros([number_of_rounds, self.n_input_cbows * 2])
            Y = np.zeros([number_of_rounds, 3])
            # create non inverse training data
            for i in range(number_of_rounds):
                x1 = [j for j in data[i:(i+self.n_input_cbows)]]
                x2= [j for j in data[i+self.n_input_cbows+hole_size:(i+hole_size + 2*self.n_input_cbows)]]
                X[i]=x1+x2
                Y[i]= one_hot(hole_size,np.arange(self.max_hole_size))

            X_f.append(X)
            Y_f.append(Y)

        # combine data for all hole sizes to one vector
        X_f=np.vstack(X_f)
        Y_f=np.vstack(Y_f)


        # [0]->[1,2], [1]->[2,3],[2]->[3,4],[n-2]->[n-1,n],

        return np.array(X_f), np.array(Y_f)

'''
def get_embeddings(reverse_dictionary):
    py2vec = Py2Vec("blog_model.json")
    embeddings = np.random.rand(len(reverse_dictionary), np.shape(py2vec["if"])[0])

    for i, key in enumerate(reverse_dictionary):
        key=string_to_token(key)["value"]
        if not any(py2vec[key]):
            print("not available:", key)
        else:
            print("available:", key)
            embeddings[i] = py2vec[key]
    return embeddings
'''

class ngram_processing:
    def __init__(self,ngram_len=3):
        self.ngrams_len=ngram_len

    # data is sequence of id of the words like ["{","var"] -> [3,5]
    def prepare_ngrams(self,data, dictionary):
        # for padding when ngrams are shorter than the stated value
        # ngrams 3 means: 2 input and 1 target output
        number_of_rounds=len(data)-self.ngrams_len+1
        X=np.zeros([number_of_rounds,self.ngrams_len-1,len(dictionary)])
        Y=np.zeros([number_of_rounds,len(dictionary)])


        # create non inverse training data
        for i in range(number_of_rounds):
            X[i]=[one_hot(i,dictionary) for i in data[i:(i+self.ngrams_len-1)]]
            Y[i]=one_hot(data[i+self.ngrams_len-1],dictionary)

        X_inv = np.zeros([number_of_rounds,self.ngrams_len-1,len(dictionary)])
        Y_inv = np.zeros([number_of_rounds,len(dictionary)])

        # create inverse training data
        for i in range(number_of_rounds):
            Y_inv[i] = one_hot(data[i],dictionary)
            X_inv[i] = [one_hot(i,dictionary) for i in data[i + 1:(i + self.ngrams_len)]]
        X_inv = [a[::-1] for a in X_inv]
        # [0]->[1,2], [1]->[2,3],[2]->[3,4],[n-2]->[n-1,n],

        return np.array(X),np.array(Y),np.array(X_inv),np.array(Y_inv)