"""Matching Net Model
As describe in the paper Matching Networks for One Shot Learning, https://arxiv.org/abs/1606.04080
Consulting the code from github repository markdtw/matching-networks"""

import tensorflow as tf


class MatchingNet:

    def __init__(self, possible_classes=5, shot=5, vector_size=1000,
                 fce=False, batch_size=32, processing_steps=10):

        """
        :param possible_classes: classes in the support set
        :param shot: pictures per class in the support set
        :param vector_size: size of output vector from the base model
        :param fce: whether to use full context embeddings
        :param batch_size: batch size
        :param processing_steps: processing steps in the fully conditional embedding f 
        """

        self.shot = shot
        self.possible_classes = possible_classes
        self.samples = shot * possible_classes
        self.fce = fce

        self.vector_size = vector_size
        self.batch_size = batch_size
        self.processing_steps = processing_steps

        self.x_hat_encode = tf.placeholder(tf.float32, shape=[None, vector_size])
        self.x_i_encode = tf.placeholder(tf.float32, shape=[None, self.samples, vector_size])
        self.y_i_ind = tf.placeholder(tf.int32, shape=[None, self.samples])
        self.y_i = tf.one_hot(self.y_i_ind, possible_classes)
        self.y_hat_ind = tf.placeholder(tf.int32, shape=[None])
        self.y_hat = tf.one_hot(self.y_hat_ind, possible_classes)

        self.acc = None
        self.loss = None

    def fce_g(self, x_i_encode):
        """
        The Fully Conditional Embedding g 
        A bidirectional LSTM
        For interpret the support set as a whole
        :param x_i_encode: support set data encoded by the base model
        :return: 
        """

        fw_cell = tf.contrib.rnn.BasicLSTMCell(self.vector_size / 2)
        bw_cell = tf.contrib.rnn.BasicLSTMCell(self.vector_size / 2)
        outputs, state_fw, state_bw = tf.contrib.rnn.static_bidirectional_rnn(fw_cell, bw_cell,
                                                                              x_i_encode,
                                                                              dtype=tf.float32)

        return tf.add(tf.stack(x_i_encode), tf.stack(outputs))

    def fce_f(self, x_hat_encode, g_embedding):
        """
        The Fully Conditional Embedding f
        :param x_hat_encode: target data encoded by the base model
        :param g_embedding: 
        :return: 
        """
        cell = tf.contrib.rnn.BasicLSTMCell(self.vector_size)
        prev_state = cell.zero_state(self.batch_size, tf.float32)

        for step in range(self.processing_steps):
            output, state = cell(x_hat_encode, prev_state)

            h_k = tf.add(output, x_hat_encode)

            content_based_attention = tf.nn.softmax(tf.multiply(prev_state[1], g_embedding))
            r_k = tf.reduce_sum(tf.multiply(content_based_attention, g_embedding), axis=0)

            prev_state = tf.contrib.rnn.LSTMStateTuple(state[0], tf.add(h_k, r_k))

        return output

    def __cosine_similarity(self, f_embedding, g_embedding):
        """
        Calculate cosine similarity
        :param f_embedding: 
        :param g_embedding: 
        :return: 
        """
        cos_sim_list = []
        for i in tf.unstack(g_embedding):
            target_normed = f_embedding
            i_normed = tf.nn.l2_normalize(i, 1)
            similarity = tf.matmul(tf.expand_dims(target_normed, 1), tf.expand_dims(i_normed, 2))
            cos_sim_list.append(tf.squeeze(similarity, [1, ]))

        cos_sim = tf.concat(axis=1, values=cos_sim_list)

        return cos_sim

    def build(self):

        x_i_encode = tf.unstack(self.x_i_encode, axis=1)

        if self.fce:
            g_embedding = self.fce_g(x_i_encode)
            f_embedding = self.fce_f(self.x_hat_encode, g_embedding)
        else:
            g_embedding = x_i_encode
            f_embedding = self.x_hat_encode

        cos_sim = self.__cosine_similarity(f_embedding, g_embedding)
        weighting = tf.nn.softmax(cos_sim)

        label_prob = tf.squeeze(tf.matmul(tf.expand_dims(weighting, 1), self.y_i))
        top_k = tf.nn.in_top_k(label_prob, self.y_hat_ind, 1)
        self.acc = tf.reduce_mean(tf.to_float(top_k))

        correct_prob = tf.reduce_sum(tf.log(tf.clip_by_value(label_prob, 1e-10, 1.0)) * self.y_hat, 1)
        self.loss = tf.reduce_mean(-correct_prob, 0)





