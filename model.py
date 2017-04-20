'''
Created on Sep 21, 2016

@author: jerrik
'''

import os
import sys
import time
import numpy as np
import tensorflow as tf
import core_rnn_cell_impl as rnn_cell
import argparse
import logging

import helper
from Config import Config
from TfUtils import entry_stop_gradients, linear
from attn_cell import AttnCell



parser = argparse.ArgumentParser(description="training options")

parser.add_argument('--load-config', action='store_true', dest='load_config', default=False)
parser.add_argument('--gpu-num', action='store', dest='gpu_num', default=0, type=int)
parser.add_argument('--train-test', action='store', dest='train_test', default='train', choices=['train', 'test'])
parser.add_argument('--weight-path', action='store', dest='weight_path', required=True)

parser.add_argument('--debug-enable', action='store_true', dest='debug_enable', default=False)

args = parser.parse_args()

class enTail(object):
    """Abstracts a Tensorflow graph for a learning task.

    We use various Model classes as usual abstractions to encapsulate tensorflow
    computational graphs. Each algorithm you will construct in this homework will
    inherit from a Model object.
    """
    def __init__(self):
        """options in this function"""
        self.config = Config()

        self.weight_Path = args.weight_path
        if args.load_config == False:
            self.config.saveConfig(self.weight_Path+'/config')
            print 'config file generated, please specify --load-config and run again'
            sys.exit()
        else:
            self.config.loadConfig(self.weight_Path+'/config')

        self.vocab = helper.Vocab(unk=True)
        self.id2tag = helper.Vocab(unk=False)
        self.vocab.load_vocab_from_file(self.config.vocab_path)
        self.id2tag.load_vocab_from_file(self.config.id2tag_path)

        self.config.class_num = len(self.id2tag)
        self.dataSet = helper.load_dataSet(self.vocab, self.id2tag, self.config.train_data,
                                           self.config.val_data, self.config.test_data)

        self.add_placeholders()
        self.embedding = self.add_embedding()
        inputs_snt1 = self.embed_lookup(self.embedding, self.ph_input_snt1, scope='lookup_snt1')
        inputs_snt2 = self.embed_lookup(self.embedding, self.ph_input_snt2, scope='lookup_snt2')

        self.logits = self.add_model(inputs_snt1, inputs_snt2,
                                     self.ph_seqLen_snt1, self.ph_seqLen_snt2)

        self.predict_prob = tf.nn.softmax(self.logits, name='predict_prob')

        self.loss = self.add_loss_op(self.logits, self.ph_input_label)
        self.train_op = self.add_train_op(self.loss)

        MyVars = [v for v in tf.trainable_variables()]
        print [v.name for v in MyVars]

    def add_placeholders(self):

        self.ph_input_label = tf.placeholder(tf.int32, (None,), name='ph_input_label')

        self.ph_input_snt1 = tf.placeholder(tf.int32, (None, None), name='ph_input_snt1')
        self.ph_input_snt2 = tf.placeholder(tf.int32, (None, None), name='ph_input_snt2')

        self.ph_seqLen_snt1 = tf.placeholder(tf.int32, (None,), name='ph_seqLen_snt1')
        self.ph_seqLen_snt2 = tf.placeholder(tf.int32, (None,), name='ph_seqLen_snt2')

    def create_feed_dict(self, data_batch):
        '''data_batch:  label_ids, snt1_matrix, snt2_matrix, snt1_len, snt2_len'''

        phs = (self.ph_input_label, self.ph_input_snt1,
               self.ph_input_snt2, self.ph_seqLen_snt1, self.ph_seqLen_snt2)
        feed_dict = dict(zip(phs, data_batch))
        return feed_dict

    def add_embedding(self):
        """Add embedding layer. that maps from vocabulary to vectors.
        inputs: a list of tensors each of which have a size of [batch_size, embed_size]
        """

        if self.config.pre_trained:
            embed = helper.readEmbedding(self.config.embed_path+str(self.config.embed_size))
            embed_matrix, valid_mask = helper.mkEmbedMatrix(embed, self.vocab.word_to_index)
            embedding = tf.Variable(embed_matrix, 'Embedding')
            embedding = entry_stop_gradients(embedding, tf.expand_dims(valid_mask, 1))
        else:
            embedding = tf.get_variable(
              'Embedding',
              [len(self.vocab), self.config.embed_size], trainable=True)
        return embedding

    def embed_lookup(self, embedding, batch_x, scope):
        inputs = tf.nn.embedding_lookup(embedding, batch_x) ## (batch_size, num_steps, embed_size)

        if self.config.cnn_after_embed:
            with tf.variable_scope('conv_'+scope):
                in_channel = self.config.embed_size
                out_channel = self.config.embed_size
                filter_shape = [3, in_channel, out_channel]
                W = tf.get_variable(name='W', shape=filter_shape)
                b = tf.get_variable(name='b', shape=[out_channel])
                conv = tf.nn.conv1d(  # size (b_sz, tstp, out_channel)
                    inputs,
                    W,
                    stride=1,
                    padding="SAME",
                    name="conv")
                inputs = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
        return inputs

###################################################################################################

    def add_model(self, input_x1, input_x2, seqLen_x1, seqLen_x2):
        '''
        dynamic_rnn(cell, inputs, sequence_length=None, initial_state=None,
                dtype=None, parallel_iterations=None, swap_memory=False,
                time_major=False, scope=None):
        '''
        with tf.variable_scope('Premise_encoder'):
            lstm_cell = rnn_cell.BasicLSTMCell(self.config.hidden_size)
            lstm_cell = rnn_cell.DropoutWrapper(lstm_cell, input_keep_prob=self.config.dropout,
                                    output_keep_prob=self.config.dropout)
            Premise_out, Premise_state = tf.nn.dynamic_rnn(cell=lstm_cell, inputs=input_x1,
                                               sequence_length=seqLen_x1,
                                               dtype=tf.float32, swap_memory=True)
        with tf.variable_scope('Hypothesis_encoder'):
            lstm_cell = rnn_cell.BasicLSTMCell(self.config.hidden_size)
            lstm_cell = rnn_cell.DropoutWrapper(lstm_cell, input_keep_prob=self.config.dropout,
                                                output_keep_prob=self.config.dropout)
            Hypo_out, Hypo_state = tf.nn.dynamic_rnn(cell=lstm_cell, inputs=input_x2,
                                               sequence_length=seqLen_x2,
                                               initial_state=Premise_state, swap_memory=True)

        def w2w_attn(Premise_out, Hypo_out, seqLen_Premise, seqLen_Hypo, scope=None):
            with tf.variable_scope(scope or 'Attn_layer'):
                attn_cell = AttnCell(self.config.hidden_size, Premise_out, seqLen_Premise)
                attn_cell = rnn_cell.DropoutWrapper(attn_cell, input_keep_prob=self.config.dropout,
                                                    output_keep_prob=self.config.dropout)

                _, r_state = tf.nn.dynamic_rnn(attn_cell, Hypo_out, seqLen_Hypo,
                                  dtype=Hypo_out.dtype, swap_memory=True)
            return r_state

        r_L = w2w_attn(Premise_out, Hypo_out, seqLen_x1, seqLen_x2, scope='w2w_attention')

        h_star = tf.tanh(linear([r_L, Hypo_state[1]],       # shape (b_sz, h_sz)
                                self.config.hidden_size, bias=False,
                                scope='linear_trans'))
        input_fully = h_star
        for i in range(self.config.fnn_layers):
            with tf.variable_scope('fully_connect_'+str(i)):
                logits = tf.contrib.layers.fully_connected(
                  input_fully, self.config.hidden_size*2, activation_fn=None)
                input_fully = tf.tanh(logits)
        with tf.name_scope('Softmax'):
            logits = tf.contrib.layers.fully_connected(
                  input_fully, self.config.class_num, activation_fn=None)
        return logits

    def add_loss_op(self, logits, labels):
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels)
        loss = tf.reduce_mean(loss)

        reg_loss = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables() if v != self.embedding])
        return loss + self.config.reg * reg_loss

    def add_train_op(self, loss):
        global_step = tf.Variable(0, name='global_step', trainable=False)
        self.learning_rate = tf.train.exponential_decay(self.config.lr, global_step,
                                                        self.config.decay_steps,
                                                        self.config.decay_rate, staircase=True)

        optimizer = tf.train.AdamOptimizer(self.learning_rate)
        train_op = optimizer.minimize(loss, global_step=global_step)
        return train_op

    def run_epoch(self, sess, data, verbose=10):
        """Runs an epoch of training.

        Trains the model for one-epoch.

        Args:
            sess: tf.Session() object
        Returns:
            average_loss: scalar. Average minibatch loss of model on epoch.
        """
        total_steps = data.size // self.config.batch_size
        total_loss = []
        collect_time = []
        for step, data_batch in enumerate(data.data_iter(batch_size=self.config.batch_size, test=False)):

            feed_dict = self.create_feed_dict(data_batch=data_batch)
            start_stamp = time.time()
            _, loss, lr = sess.run([self.train_op, self.loss, self.learning_rate], feed_dict=feed_dict)
            end_stamp = time.time()

            collect_time.append(end_stamp-start_stamp)
            total_loss.append(loss)

            if verbose and step % verbose == 0:
                sys.stdout.write('\r%d / %d : loss = %f, %.3fs/iter, lr = %f' % (
                    step, total_steps, np.mean(total_loss[-verbose:]), np.mean(collect_time), lr))
                collect_time = []
                sys.stdout.flush()
        return np.mean(total_loss)

    def fit(self, sess, data, verbose=10):
        """
        Fit the model.

        Args:
            sess: tf.Session() object
        Returns:
            average_loss: scalar. Average minibatch loss of model on epoch.
        """
        total_steps = data.size // self.config.batch_size
        total_loss = []
        collect_time=[]
        for step, data_batch in enumerate(data.data_iter(batch_size=self.config.batch_size, test=True)):
            feed_dict = self.create_feed_dict(data_batch=data_batch)
            start_stamp = time.time()
            loss = sess.run(self.loss, feed_dict=feed_dict)
            end_stamp = time.time()
            collect_time.append(end_stamp - start_stamp)
            total_loss.append(loss)

            if verbose and step % verbose == 0:
                sys.stdout.write('\r%d / %d : loss = %f,  %.3fs/iter' % (
                    step, total_steps, np.mean(total_loss[-verbose:]), np.mean(collect_time)))
                collect_time = []
                sys.stdout.flush()
        print '\n'
        return np.mean(total_loss)

    def predict(self, sess, data, verbose=10):
        """
        Args:
            sess: tf.Session() object
        Returns:
            average_loss: scalar. Average minibatch loss of model on epoch.
        """

        total_steps = data.size // self.config.batch_size
        collect_predict = []
        label_id = []
        collect_time = []
        for step, data_batch in enumerate(data.data_iter(batch_size=self.config.batch_size, test=True)):
            feed_dict = self.create_feed_dict(data_batch=data_batch)
            start_stamp = time.time()
            predict_prob = sess.run(self.predict_prob, feed_dict=feed_dict)
            end_stamp = time.time()
            collect_time.append(end_stamp - start_stamp)

            collect_predict.append(predict_prob)
            label_id += data_batch[0]
            if verbose and step % verbose == 0:
                sys.stdout.write('\r%d / %d : , %.3fs/iter' % (
                    step, total_steps, np.mean(collect_time)))
                collect_time = []
                sys.stdout.flush()
        print '\n'
        res_prob = np.concatenate(collect_predict, axis=0)
        return res_prob, label_id

def test_case(sess, model, data, onset='VALIDATION'):
    print '#'*20, 'ON '+onset+' SET START ', '#'*20
    loss = model.fit(sess, data=data)
    pred_prob, label = model.predict(sess, data=data)

    pred = helper.pred_from_prob_single(pred_prob) #(data_num, )
    prec, recall, overall_prec, overall_recall, _ = helper.calculate_confusion_single(
                                            pred, label, model.config.class_num)
    helper.print_confusion_single(prec, recall, overall_prec, overall_recall, model.id2tag.index_to_word)
    accuracy = helper.calculate_accuracy_single(pred, label)

    print 'Overall '+onset+' accuracy is: {}'.format(accuracy)
    logging.info('Overall '+onset+' accuracy is: {}'.format(accuracy))

    print 'Overall ' + onset + ' loss is: {}'.format(loss)
    logging.info('Overall ' + onset + ' loss is: {}'.format(loss))
    print '#'*20, 'ON '+onset+' SET END ', '#'*20
    return accuracy

def train_run():
    logging.info('Training start')
    with tf.Graph().as_default():
        with tf.device("/gpu:" + str(args.gpu_num)):
            entail_model = enTail()
        saver = tf.train.Saver()

        config=tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        with tf.Session(config=config) as sess:

            best_accuracy = 0
            best_val_epoch = 0
            sess.run(tf.global_variables_initializer())

            for epoch in range(entail_model.config.max_epochs):
                print "="*20+"Epoch ", epoch, "="*20
                loss = entail_model.run_epoch(sess, data=entail_model.dataSet.train)
                print
                print "Mean loss in this epoch is: ", loss
                logging.info("Mean loss in {}th epoch is: {}".format(epoch, loss) )
                print '='*50

                val_accuracy = test_case(sess, entail_model, entail_model.dataSet.valid, onset='VALIDATION')

                if best_accuracy < val_accuracy:
                    best_accuracy = val_accuracy
                    best_val_epoch = epoch
                    if not os.path.exists(entail_model.weight_Path):
                        os.makedirs(entail_model.weight_Path)

                    saver.save(sess, entail_model.weight_Path+'/classifier.weights')
                if epoch - best_val_epoch > entail_model.config.early_stopping:
                    logging.info("Normal Early stop")
                    break
    logging.info("Training complete")

def test_run():

    with tf.Graph().as_default():
        with tf.device("/gpu:" + str(args.gpu_num)):   #gpu_num options
            entail_model = enTail()
        saver = tf.train.Saver()

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        with tf.Session(config=config) as sess:
            sess.run(tf.global_variables_initializer())
            saver.restore(sess, entail_model.weight_Path+'/classifier.weights')

            test_case(sess, entail_model, entail_model.dataSet.test, onset='TEST')

def main(_):
    if not os.path.exists(args.weight_path):
        os.makedirs(args.weight_path)
    logFile = args.weight_path+'/run.log'

    if args.train_test == "train":

        try:
            os.remove(logFile)
        except OSError:
            pass
        logging.basicConfig(filename=logFile, format='%(levelname)s %(asctime)s %(message)s', level=logging.INFO)
        train_run()
    else:
        logging.basicConfig(filename=logFile, format='%(levelname)s %(asctime)s %(message)s', level=logging.INFO)
        test_run()

if __name__ == '__main__':
    tf.app.run()
