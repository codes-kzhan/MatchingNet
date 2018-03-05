import tensorflow as tf
import mobilenet_v1


class MatchingNet:

    def __init__(self, pic_size, shot, possible_classes, model_type='MNIST',
                 channel=3, fce=False, batch_size=32, processing_steps=10):

        self.pic_size = pic_size
        self.shot = shot
        self.possible_classes = possible_classes
        self.samples = shot * possible_classes
        self.channel = channel
        self.fce = fce

        self.model_type = model_type

        self.vector_size = None
        self.batch_size = batch_size
        self.processing_steps = processing_steps

        self.x_hat = tf.placeholder(tf.float32, shape=[None, pic_size, pic_size, channel])
        self.x_i = tf.placeholder(tf.float32, shape=[None, self.samples, pic_size, pic_size, channel])
        self.y_i_ind = tf.placeholder(tf.int32, shape=[None, self.samples])
        self.y_i = tf.one_hot(self.y_i_ind, possible_classes)
        self.y_hat_ind = tf.placeholder(tf.int32, shape=[None])
        self.y_hat = tf.one_hot(self.y_hat_ind, possible_classes)

        self.acc = None
        self.loss = None

    def base_model(self, features):

        if self.model_type == 'MNIST':

            with tf.variable_scope('', reuse=tf.AUTO_REUSE):
                input_layer = tf.reshape(features, [-1, self.pic_size, self.pic_size, self.channel])

                conv1 = tf.layers.conv2d(
                    inputs=input_layer,
                    filters=32,
                    kernel_size=[5, 5],
                    padding="same",
                    activation=tf.nn.relu)

                pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

                conv2 = tf.layers.conv2d(
                    inputs=pool1,
                    filters=64,
                    kernel_size=[5, 5],
                    padding="same",
                    activation=tf.nn.relu)

                pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

                pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 64])

                dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)

                return dense

        elif self.model_type == 'MOBILENET':

            logits, _ = mobilenet_v1.mobilenet_v1(features, prediction_fn=None,
                                                  reuse=tf.AUTO_REUSE, scope='FeatureExtractor/MobilenetV1')

            return logits

    def fce_g(self, x_i_encode):
        fw_cell = tf.contrib.rnn.BasicLSTMCell(self.vector_size / 2)
        bw_cell = tf.contrib.rnn.BasicLSTMCell(self.vector_size / 2)
        outputs, state_fw, state_bw = tf.contrib.rnn.static_bidirectional_rnn(fw_cell, bw_cell,
                                                                              x_i_encode, dtype=tf.float32)

        return tf.add(tf.stack(x_i_encode), tf.stack(outputs))

    def fce_f(self, x_hat_encode, g_embedding):
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

        cos_sim_list = []
        for i in tf.unstack(g_embedding):
            target_normed = f_embedding
            i_normed = tf.nn.l2_normalize(i, 1)
            similarity = tf.matmul(tf.expand_dims(target_normed, 1), tf.expand_dims(i_normed, 2))
            cos_sim_list.append(tf.squeeze(similarity, [1, ]))

        cos_sim = tf.concat(axis=1, values=cos_sim_list)

        return cos_sim

    def build(self):

        x_hat_encode = self.base_model(self.x_hat)
        x_i_encode = [self.base_model(i) for i in tf.unstack(self.x_i, axis=1)]

        self.vector_size = int(x_hat_encode.shape[1])

        if self.fce:
            g_embedding = self.fce_g(x_i_encode)
            f_embedding = self.fce_f(x_hat_encode, g_embedding)
        else:
            g_embedding = x_i_encode
            f_embedding = x_hat_encode

        cos_sim = self.__cosine_similarity(f_embedding, g_embedding)
        weighting = tf.nn.softmax(cos_sim)

        label_prob = tf.squeeze(tf.matmul(tf.expand_dims(weighting, 1), self.y_i))
        top_k = tf.nn.in_top_k(label_prob, self.y_hat_ind, 1)
        self.acc = tf.reduce_mean(tf.to_float(top_k))

        correct_prob = tf.reduce_sum(tf.log(tf.clip_by_value(label_prob, 1e-10, 1.0)) * self.y_hat, 1)
        self.loss = tf.reduce_mean(-correct_prob, 0)





