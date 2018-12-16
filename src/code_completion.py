
from __future__ import print_function


from src.utils import *
from tqdm import tqdm
from src.neural_networks import RNN, DNN

class Code_Completion:
    def __init__(self):
        self.session=tf.Session()
        # settings from json file
        self.config=getJsonDataFromConfigFile("config.json")


        self.lOSS_LABEL_HOLE_PRED="Hole Prediction with CBOWS"
        self.lOSS_LABEL_LSTM_FORWARD_PRED="Forward LSTM"
        self.lOSS_LABEL_LSTM_BACKWARD_PRED="Backward LSTM"


        self.n_input_cbows=self.config["cbow"]["length"]
        self.max_hole_size=self.config["cbow"]["max_hole_size"]
        # trigram is n_input=2
        self.ngram_len=self.config["ngram"]["length"]
        # Parameters
        #dropout value
        self.keep_prob=self.config["DNN"]["dropout"]
        self.train_episodes = self.config["alg"]["train_episodes"]
        self.batch_size=self.config["alg"]["batch_size"]
        self.display_step = self.config["alg"]["display_step"]

    def create_network_and_prepare_word_processing(self,load_model=False):

        cbow_proc=cbow_processing(max_hole_size=self.max_hole_size, n_input_cbows = self.n_input_cbows)
        ngram_proc=ngram_processing(ngram_len=self.ngram_len)
        self.X_cbows,self.Y_cbows=cbow_proc.prepare_cbows(self.training_data,self.reverse_dictionary)

        self.X,self.Y,self.X_inv,self.Y_inv=ngram_proc.prepare_ngrams(self.training_data,self.reverse_dictionary)
        vocab_size = len(self.dictionary)

        #embeddings=get_embeddings(self.reverse_dictionary)
        self.model_forward_rnn_file_name=self.model_file_name + "_forward_rnn"
        self.model_backward_rnn_file_name=self.model_file_name + "_backward_rnn"
        self.model_dnn_file_name= self.model_file_name + "_dnn"
        if load_model:
            self.model_forward_rnn_file=load_model_from_file_for_keras(self.model_file_name + "_forward_rnn")
            self.model_backward_rnn_file=load_model_from_file_for_keras(self.model_file_name + "_backward_rnn")
        else:
            self.model_forward_rnn_file = None
            self.model_backward_rnn_file = None

        # initialize all three NN
        self.Pred_forward = RNN(vocab_size,self.model_forward_rnn_file)
        self.Pred_backward=RNN(vocab_size,self.model_backward_rnn_file)
        self.Pred_hole_size = DNN(self.n_input_cbows,self.max_hole_size)

        # instantiate all NN
        self.pred_forward=self.Pred_forward("forward")
        self.pred_backward=self.Pred_backward("backward")
        self.pred_hole = self.Pred_hole_size()


        # Initializing the variables
        self.init = tf.global_variables_initializer()
        # for storing the model to file
        self.saver = tf.train.Saver()
        if load_model:
            self.session.run(self.init)
            self.session=load_model_from_file(self.session,self.model_dnn_file_name)

    def prepare_data(self, token_lists):
        # encode tokens into one-hot vectors
        all_token_strings = set()
        for token_list in token_lists:
            for token in token_list:
                all_token_strings.add(token_to_string(token))
        all_token_strings = list(all_token_strings)
        all_token_strings.sort()
        self.string_to_number = dict()
        self.number_to_string = dict()
        max_number = 0
        for token_string in all_token_strings:
            self.string_to_number[token_string] = max_number
            self.number_to_string[max_number] = token_string
            max_number += 1

        word2idx=[]
        for token_list in token_lists:
            for idx, token in enumerate(token_list):
                word2idx.append(self.string_to_number[token_to_string(token)])
        # dict-type of form: number->string,i.e. dictionary[80]->"Punctuator"
        self.dictionary=self.number_to_string
        # dict-type of form: string->number,i.e. "Punctuator" -> dictionary[80]
        self.reverse_dictionary=self.string_to_number
        self.training_data=word2idx

    def load(self, token_lists, model_file_name):
        self.model_file_name = model_file_name
        self.prepare_data(token_lists)
        self.create_network_and_prepare_word_processing(load_model=True)
        # load DNN model for hole size prediction implemented with pure tensorflow
        #load_model_from_file(self.session, self.model_dnn_file_name, self.saver)

    def train(self, token_lists, model_file_name):
        self.model_file_name=model_file_name
        self.prepare_data(token_lists)
        self.create_network_and_prepare_word_processing(load_model=False)
        self.fit()
        #self.create_network()
        #self.model.fit(xs, ys, n_epoch=1, batch_size=1024, show_metric=True)
        #self.model.save(model_file)



    def query_hole_size(self,cbows_data):
        # x_cbows = np.reshape(np.array(symbols_in_keys), [-1, n_input, 1])
        pred=self.Pred_hole_size.predict_hole_size(cbows_data,self.session,keep_prob=1.0)
        return pred

    def query_suffix(self,prefix_ngram):
        one_hot_ngram=[one_hot(int(i),self.reverse_dictionary) for i in np.squeeze(prefix_ngram)]
        onehot_pred=self.Pred_forward.predict(one_hot_ngram,self.session)
        return onehot_pred

    def query_prefix(self,suffix_ngram):
        one_hot_ngram = [one_hot(int(i), self.reverse_dictionary) for i in np.squeeze(suffix_ngram)]
        onehot_pred = self.Pred_forward.predict(one_hot_ngram, self.session)
        return onehot_pred

    # the given prefix and suffix has a long length, so it has to be cut to the size of the
    # used ngrams size during training, every item in their array has a type-value-pair as dict
    def query(self,prefix, suffix):
            # fill ngram-vectors with zeros as padding
            prefix_ngram=np.zeros(self.ngram_len-1)
            suffix_ngram=np.zeros(self.ngram_len-1)
            # ngram-preperation
            prefix_ngram[:min(self.ngram_len-1,np.shape(prefix)[0])]=get_indexed_tokens_by_dict_data(-self.ngram_len+1,None,prefix,self.reverse_dictionary)
            suffix_ngram[:min(self.ngram_len-1,np.shape(suffix)[0])]=get_indexed_tokens_by_dict_data(None,self.ngram_len-1,suffix,self.reverse_dictionary)
            # get the prefix data of the hole and then the suffix data to get the cbow of it
            # padding with zeros to fill to input size of the neural network
            prefix_cbows=np.zeros(self.n_input_cbows)
            suffix_cbows=np.zeros(self.n_input_cbows)
            prefix_cbows[-min(self.n_input_cbows,np.shape(prefix)[0]):]=get_indexed_tokens_by_dict_data(-self.n_input_cbows,None,prefix,self.reverse_dictionary)
            suffix_cbows[:min(self.n_input_cbows,np.shape(suffix)[0])]=get_indexed_tokens_by_dict_data(None,self.n_input_cbows,suffix,self.reverse_dictionary)
            # concetenate cbow prefix and suffix
            cbows_data=np.hstack((prefix_cbows,suffix_cbows))
            cbows_data=np.reshape(np.array(cbows_data),(-1,2*self.n_input_cbows))
            prefix_ngram=np.reshape(prefix_ngram,(-1,self.ngram_len-1))
            suffix_ngram=np.reshape(suffix_ngram,(-1,self.ngram_len-1))

            # cbow
            # 1st predict hole size
            # variable hole_size is an array of size of the size of the hole with probabilities of the hole-size
            hole_size=self.Pred_hole_size.predict_hole_size(cbows_data,self.session,keep_prob=1.0)

            hole_size=np.argmax(hole_size)

            # 2nd if hole-size 1-2 then both NN predict the token after and before it respectively
            suffix_predictions=[]
            prefix_predictions=[]
            prev_prefix =prefix_ngram
            prev_suffix=suffix_ngram
            cur_suffix=cur_prefix=[]
            for i in range(int(hole_size)+1):
                # predict next token as one-hot by previous token
                # pass only n-grams of size ngram_len to query, i.e. of the sequence [23,45,21]
                # only the last two tokens are extracted in trigram
                cur_suffix_one_hot=self.query_suffix(prev_prefix[:,-self.ngram_len+1:])
                cur_prefix_one_hot=self.query_prefix(prev_suffix[:,-self.ngram_len+1:])
                # convert one-hot to token as number, because it is needed as argument for RNN
                cur_suffix_idx_number=np.argmax(cur_suffix_one_hot)
                cur_prefix_idx_number=np.argmax(cur_prefix_one_hot)
                # add predicted token to array
                suffix_predictions.append(cur_suffix_one_hot)
                prefix_predictions.append(cur_prefix_one_hot)
                # assign predicted next token as previous token to predict next token,
                # concetenate the whole sequence
                prev_suffix=np.expand_dims(np.hstack((np.squeeze(prev_suffix),cur_suffix_idx_number)),axis=0)
                prev_prefix=np.expand_dims(np.hstack((np.squeeze(prev_prefix),cur_prefix_idx_number)),axis=0)
            completion=self.predict_completion(np.array(prefix_predictions),np.array(suffix_predictions))
            return completion
            # if hole-size=3 then both NN predict 2 tokens after and before it respectively
            # and second prediction is choicen by taking the token predicted by both NN

    def predict_completion(self,prefix_predictions, suffix_predictions):
        # prefix_predictions and suffix_predictions are an array of one-hot values of the predicted tokens
        prefix_predictions=prefix_predictions[::-1]
        multiplied_probabilities=prefix_predictions*suffix_predictions
        # index of highest common probability of prefix- and suffix-prediction
        idx_of_highest_probability=[np.argmax(i) for i in multiplied_probabilities]

        # get tokens as strings of tokens with highest probability
        # return the tokens as an dict-object of format {type, value}
        predictions=[string_to_token(self.dictionary[i]) for i in idx_of_highest_probability]
        return predictions



    def train_forward_LSTM(self,batch_data,batch_labels):
        model=self.Pred_forward.get_accuracy(batch_data=batch_data,batch_labels=batch_labels,sess=self.session)
        save_model_to_file_for_keras(model, self.model_forward_rnn_file_name)
        print("saving forward model forward to ", self.model_forward_rnn_file_name)

    def train_backward_LSTM(self,batch_data,batch_labels):
        # just use inversed ngrams and inversed output and reuse the previous LSTM-Neural Network
        # optimize the gradients
        model=self.Pred_backward.get_accuracy(batch_data=batch_data,batch_labels=batch_labels,sess=self.session)
        save_model_to_file_for_keras(model, self.model_backward_rnn_file_name)
        print("saving backward model backward to ", self.model_file_name)

    def train_hole_size_with_CBOWS(self,batch_data,batch_labels):
        # optimize the gradients
        loss_cbows=self.Pred_hole_size.optimize(batch_data=batch_data,batch_labels=batch_labels,sess=self.session,keep_prob=self.keep_prob)
        # get predicted hole as onehot i.e. [0,1,0], accuracy and an array with bools about the correct prediction i.e. [False, True,True...]
        onehot_pred_cbows_eval,onehot_pred_cbows,accuracy = self.Pred_hole_size.get_accuracy(batch_data=batch_data,batch_labels=batch_labels,sess=self.session,keep_prob=self.keep_prob)
        return onehot_pred_cbows,onehot_pred_cbows_eval, accuracy, loss_cbows

    def get_minibatches(self,inputs, targets, batchsize, shuffle=True):
        if shuffle:
            indices = np.arange(inputs.shape[0])
            np.random.shuffle(indices)
        for start_idx in range(0, inputs.shape[0] - batchsize + 1, batchsize):
            if shuffle:
                slices = indices[start_idx:start_idx + batchsize]
            else:
                slices = slice(start_idx, start_idx + batchsize)
            yield inputs[slices], targets[slices]



    def fit(self):
            self.session.run(self.init)
            # to activate the individual prediction processes
            cbow_training = True
            forward_training = True
            backward_training = True

            ################## cbows training #########
            if cbow_training:
                # run through a number of episodes
                for _ in tqdm(range(self.train_episodes)):
                    # loop over mini batches
                    self.batch_size=self.config["alg"]["batch_size"]
                    # iterate over all minibatches once. To avoid overfitting shuffling is recommended
                    for iter,batch_cbows in enumerate(self.get_minibatches(self.X_cbows, self.Y_cbows, self.batch_size, shuffle=True)):
                        batch_data_cbows, batch_labels_cbows = batch_cbows

                        onehot_pred_cbows,onehot_pred_cbows_eval,acc_cbows, loss_cbows = self.train_hole_size_with_CBOWS(batch_data_cbows,batch_labels_cbows)
                        onehot_pred_cbows=np.reshape(onehot_pred_cbows,[-1,self.n_input_cbows])
                        onehot_pred_cbows=[one_hot([int(self.session.run(tf.argmax(i)))],np.arange(self.max_hole_size)) for i in onehot_pred_cbows]

                        nrtrue=len([i for i in onehot_pred_cbows_eval if i==True])
                        nrtotal=len(onehot_pred_cbows)

                        #acc=nrtrue/nrtotal*100
                        #print("Accuracy DNN=",acc)
                        # after every 20 iterations save model
                        if iter % 20:
                            save_model_to_file(self.session, self.model_dnn_file_name)
                    #loss_total_cbows += loss_cbows
                    #acc_total_cbows += acc_cbows


                    #print("Accuracy DNN total=",acc_total_cbows)
                    #print("Loss DNN total=",loss_total_cbows)

            ################## forward LSTM #########
            if forward_training:
                self.train_forward_LSTM(self.X,self.Y)

            ################# inverse LSTM ################
            # just use inversed ngrams and inversed output and reuse the previous LSTM-Neural Network
            if backward_training:
                self.train_backward_LSTM(self.X_inv, self.Y_inv)

