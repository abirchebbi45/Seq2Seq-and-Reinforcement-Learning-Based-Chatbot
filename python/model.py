import tensorflow as tf
import numpy as np

class Seq2Seq_chatbot():
    def __init__(self, dim_wordvec, n_words, dim_hidden, batch_size, n_encode_lstm_step, n_decode_lstm_step, bias_init_vector=None, lr=0.0001):
        self.dim_wordvec = dim_wordvec
        self.dim_hidden = dim_hidden
        self.batch_size = batch_size
        self.n_words = n_words
        self.n_encode_lstm_step = n_encode_lstm_step
        self.n_decode_lstm_step = n_decode_lstm_step
        self.lr = lr

        with tf.device("/cpu:0"):
            self.Wemb = tf.Variable(tf.random_uniform([n_words, dim_hidden], -0.1, 0.1), name='Wemb')

        self.lstm1 = tf.contrib.rnn.BasicLSTMCell(dim_hidden, state_is_tuple=False)
        self.lstm2 = tf.contrib.rnn.BasicLSTMCell(dim_hidden, state_is_tuple=False)

        self.encode_vector_W = tf.Variable(tf.random_uniform([dim_wordvec, dim_hidden], -0.1, 0.1), name='encode_vector_W')
        self.encode_vector_b = tf.Variable(tf.zeros([dim_hidden]), name='encode_vector_b')

        self.embed_word_W = tf.Variable(tf.random_uniform([dim_hidden, n_words], -0.1, 0.1), name='embed_word_W')
        if bias_init_vector is not None:
            self.embed_word_b = tf.Variable(bias_init_vector.astype(np.float32), name='embed_word_b')
        else:
            self.embed_word_b = tf.Variable(tf.zeros([n_words]), name='embed_word_b')