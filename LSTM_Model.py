import tensorflow as tf
import DataHolder
import numpy as np


def Fully_Connected(inp, output, name, activation, reuse=False):
    h = tf.contrib.layers.fully_connected(
        inputs=inp,
        num_outputs=output,
        activation_fn=activation,
        weights_initializer=tf.contrib.layers.xavier_initializer(),
        weights_regularizer=tf.contrib.layers.l2_regularizer(scale=3e-7),
        biases_initializer=tf.constant_initializer(3e-7),
        scope=name,
        reuse=reuse
    )

    return h


def seq_length(sequence):
    used = tf.sign(tf.reduce_max(tf.abs(sequence), reduction_indices=2))
    length = tf.reduce_sum(used, reduction_indices=1)
    length = tf.cast(length, tf.int32)
    return length


def create_initializer(initializer_range=0.02):
    """Creates a `truncated_normal_initializer` with the given range."""
    return tf.truncated_normal_initializer(stddev=initializer_range)


class LSTM_Model:
    def __init__(self):
        self.data_holder = DataHolder.DataHolder()

        self.X = tf.placeholder(shape=[None, None], dtype=np.int32)
        self.Y = tf.placeholder(shape=[None, 2], dtype=np.int32)

    def forward(self):
        pos_embedding = tf.get_variable(
            name="word_embedding",
            shape=[60000, 200],
            initializer=create_initializer(0.02))

        sequence = tf.nn.embedding_lookup(pos_embedding, self.X)

        cell_fw_L1 = tf.nn.rnn_cell.GRUCell(200)
        cell_fw_L2 = tf.nn.rnn_cell.GRUCell(150)

        with tf.variable_scope("rnn_model1"):
            output_fw, state_fw = tf.nn.dynamic_rnn(
                inputs=sequence, cell=cell_fw_L1,
                sequence_length=seq_length(sequence), dtype=tf.float32, time_major=False)
        with tf.variable_scope("rnn_model2"):
            output_fw, state_fw = tf.nn.dynamic_rnn(
                inputs=output_fw, cell=cell_fw_L2,
                sequence_length=seq_length(sequence), dtype=tf.float32, time_major=False)

        with tf.variable_scope("prediction"):
            hidden_states = Fully_Connected(state_fw, output=100, name='hidden_layer1', activation=tf.nn.tanh,
                                             reuse=False)

            hidden_states = Fully_Connected(hidden_states, output=50, name='hidden_layer2', activation=tf.nn.tanh,
                                             reuse=False)

            prediction = Fully_Connected(hidden_states, output=2, name='prediction_layer', activation=tf.nn.sigmoid,
                                            reuse=False)

        return prediction

    def training(self, epo, l2_norm=True, continue_training=False):
        #실험하는 컴퓨터 환경에 맞게 수정하세요
        save_path = 'C:\\Users\\USER\\Desktop\\rating_movies\\my_model'

        with tf.Session() as sess:
            prediction = self.forward()

            total_loss = tf.nn.softmax_cross_entropy_with_logits(labels=self.Y, logits=prediction)

            regularizer = tf.contrib.layers.l2_regularizer(scale=3e-7)
            if l2_norm is not None:
                variables = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
                l2_loss = tf.contrib.layers.apply_regularization(regularizer, variables)
                total_loss += l2_loss

            total_loss = tf.reduce_mean(total_loss)

            learning_rate = 0.001

            optimizer = tf.train.AdamOptimizer(learning_rate).minimize(total_loss)

            sess.run(tf.initialize_all_variables())

            if continue_training is True:
                print('model restoring!')
                saver = tf.train.Saver()
                saver.restore(sess, save_path)

            for i in range(epo):
                sequence1, label = self.data_holder.next_random_batch()
                training_feed_dict = {self.X: sequence1, self.Y: label}

                _, loss_value = sess.run([optimizer, total_loss], feed_dict=training_feed_dict)

                if i % 100 == 0:
                    print(i, loss_value)

                if i % 1000 == 0:
                    saver = tf.train.Saver()
                    saver.save(sess, save_path)
                    print('saved!')
                    print('loss:', loss_value)

            saver = tf.train.Saver()
            saver.save(sess, save_path)

    def evaluation(self):
        # 실험하는 컴퓨터 환경에 맞게 수정하세요
        save_path = 'C:\\Users\\USER\\Desktop\\rating_movies\\my_model'

        with tf.Session() as sess:
            prediction = self.forward()
            prediction_idx = tf.argmax(prediction, axis=1)

            sess.run(tf.initialize_all_variables())

            saver = tf.train.Saver()
            saver.restore(sess, save_path)

            cor = 0
            cnt = 0

            for i in range(50):
            #while(True):
                check, sequence1, label = self.data_holder.next_test_batch()

                if check is False:
                    break

                training_feed_dict = {self.X: sequence1}

                result = sess.run(prediction_idx, feed_dict=training_feed_dict)

                for i in range(self.data_holder.Batch_Size):
                    if result[i] == label[i]:
                        cor += 1
                    cnt += 1

        print('학번과 이름을 출력하세요', cor, '/', cnt)