import tensorflow as tf
import numpy as np
import random
import time
import sys
import matplotlib.pyplot as plt

## RNN with num_layers LSTM layers and a fully-connected output layer
## The network allows for a dynamic number of iterations, depending on the inputs it receives.
##
##    out   (fc layer; out_size)
##     ^
##    lstm
##     ^
##    lstm  (lstm size)
##     ^
##     in   (in_size)
class ModelNetwork:
    def __init__(self, in_size, lstm_size, num_layers, out_size, session, learning_rate=0.003, name="rnn"):
        self.scope = name

        self.in_size = in_size
        self.lstm_size = lstm_size
        self.num_layers = num_layers
        self.out_size = out_size

        self.session = session

        self.learning_rate = tf.constant(learning_rate)

        # Last state of LSTM, used when running the network in TEST mode
        self.lstm_last_state = np.zeros((self.num_layers * 2 * self.lstm_size,))

        with tf.variable_scope(self.scope):
            ## (batch_size, timesteps, in_size)
            self.xinput = tf.placeholder(tf.float32, shape=(None, None, self.in_size), name="xinput")
            self.lstm_init_value = tf.placeholder(tf.float32, shape=(None, self.num_layers * 2 * self.lstm_size),
                                                  name="lstm_init_value")

            # LSTM
            self.lstm_cells = [tf.contrib.rnn.BasicLSTMCell(self.lstm_size, forget_bias=1.0, state_is_tuple=False) for i
                               in range(self.num_layers)]
            self.lstm = tf.contrib.rnn.MultiRNNCell(self.lstm_cells, state_is_tuple=False)

            # Iteratively compute output of recurrent network
            outputs, self.lstm_new_state = tf.nn.dynamic_rnn(self.lstm, self.xinput, initial_state=self.lstm_init_value,
                                                             dtype=tf.float32)

            # Linear activation (Fully Connected layer on top of the LSTM net)
            self.rnn_out_W = tf.Variable(tf.random_normal((self.lstm_size, self.out_size), stddev=0.01))
            self.rnn_out_B = tf.Variable(tf.random_normal((self.out_size,), stddev=0.01))

            outputs_reshaped = tf.reshape(outputs, [-1, self.lstm_size])
            network_output = (tf.matmul(outputs_reshaped, self.rnn_out_W) + self.rnn_out_B)

            batch_time_shape = tf.shape(outputs)
            self.final_outputs = tf.reshape(tf.nn.softmax(network_output),
                                            (batch_time_shape[0], batch_time_shape[1], self.out_size))

            ## Training: provide target outputs for supervised training.
            self.y_batch = tf.placeholder(tf.float32, (None, None, self.out_size))
            y_batch_long = tf.reshape(self.y_batch, [-1, self.out_size])

            self.cost = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(logits=network_output, labels=y_batch_long))
            self.train_op = tf.train.RMSPropOptimizer(self.learning_rate, 0.9).minimize(self.cost)

    ## Input: X is a single element, not a list!
    def run_step(self, x, init_zero_state=True):
        ## Reset the initial state of the network.
        if init_zero_state:
            init_value = np.zeros((self.num_layers * 2 * self.lstm_size,))
        else:
            init_value = self.lstm_last_state

        out, next_lstm_state = self.session.run([self.final_outputs, self.lstm_new_state],
                                                feed_dict={self.xinput: [x], self.lstm_init_value: [init_value]})

        self.lstm_last_state = next_lstm_state[0]

        return out[0][0]

    ## xbatch must be (batch_size, timesteps, input_size)
    ## ybatch must be (batch_size, timesteps, output_size)
    def train_batch(self, xbatch, ybatch):
        init_value = np.zeros((xbatch.shape[0], self.num_layers * 2 * self.lstm_size))

        cost, _ = self.session.run([self.cost, self.train_op], feed_dict={self.xinput: xbatch, self.y_batch: ybatch,
                                                                          self.lstm_init_value: init_value})

        return cost

    def test_batch(self, xbatch, ybatch):
        init_value = np.zeros((xbatch.shape[0], self.num_layers * 2 * self.lstm_size))

        cost = self.session.run([self.cost], feed_dict={self.xinput: xbatch, self.y_batch: ybatch,
                                                                          self.lstm_init_value: init_value})

        return cost


def convert_to_one_hot(data, vocab_size):
    one_hot = np.zeros((len(data), vocab_size))
    for i in range(len(data)):
        one_hot[i][data[i]] = 1
    return one_hot


def run(train_data, test_data, prefix, vocab_size, gen_length, ckpt_file=""):
    data = convert_to_one_hot(train_data, vocab_size)
    test = convert_to_one_hot(test_data, vocab_size)

    in_size = out_size = vocab_size
    lstm_size = 16  # 128
    num_layers = 2
    batch_size = 8  # 128
    time_steps = 28  # 50
    test_batch_size = 3

    NUM_TRAIN_BATCHES = 1000

    ## Initialize the network
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.InteractiveSession(config=config)

    net = ModelNetwork(in_size=in_size,
                       lstm_size=lstm_size,
                       num_layers=num_layers,
                       out_size=out_size,
                       session=sess,
                       learning_rate=0.003,
                       name="char_rnn_network")

    sess.run(tf.global_variables_initializer())

    saver = tf.train.Saver(tf.global_variables())

    ## 1) TRAIN THE NETWORK
    if ckpt_file == "":
        last_time = time.time()

        batch = np.zeros((batch_size, time_steps, in_size))
        batch_y = np.zeros((batch_size, time_steps, in_size))

        test_batch = np.zeros((test_batch_size, time_steps, in_size))
        test_batch_y = np.zeros((test_batch_size, time_steps, in_size))
        #Start training at random spot as long as its at the start of a measure (4 beats).
        possible_batch_ids = [i*4 for i in range((data.shape[0] - time_steps)//4 - 1)]
        possible_batch_ids_test = [i*4 for i in range((test.shape[0] - time_steps)//4 - 1)]
        epochs = [20*i for i in range(NUM_TRAIN_BATCHES//20)]
        test_costs = []
        train_costs = []
        for i in range(NUM_TRAIN_BATCHES):
            # Sample time_steps consecutive samples
            batch_id = random.sample(possible_batch_ids, batch_size)
            batch_id_test = random.sample(possible_batch_ids_test, test_batch_size)

            for j in range(time_steps):
                ind1 = [k + j for k in batch_id]
                ind2 = [k + j + 1 for k in batch_id]
                batch[:, j, :] = data[ind1, :]
                batch_y[:, j, :] = data[ind2, :]

                ind1_test = [k + j for k in batch_id_test]
                ind2_test = [k + j + 1 for k in batch_id_test]
                test_batch[:, j, :] = test[ind1_test, :]
                test_batch_y[:, j, :] = test[ind2_test, :]

            cst = net.train_batch(batch, batch_y)

            if (i % 20) == 0:
                new_time = time.time()
                diff = new_time - last_time
                last_time = new_time

                test_cost = net.test_batch(test_batch,test_batch_y)
                print("batch: ", i, "   loss: ", cst, "test loss: ", test_cost, "   speed: ", (20.0 / diff), " batches / s")
                test_costs.append(test_cost)
                train_costs.append(cst)


        saver.save(sess, "saves/model.ckpt")
        test_plt = plt.plot(epochs, test_costs, label='Test Set')
        train_plt = plt.plot(epochs,train_costs, label='Train Set')
        plt.xlabel('Number of Epochs')
        plt.ylabel('Cross Entropy Loss')
        plt.legend(loc='best')
        plt.show()

    ## 2) GENERATE LEN_TEST_TEXT CHARACTERS USING THE TRAINED NETWORK

    if ckpt_file != "":
        saver.restore(sess, ckpt_file)

    for i in range(len(prefix)):
        out = net.run_step(convert_to_one_hot([prefix[i]], vocab_size), i == 0)

    gen_str = []
    for i in range(gen_length):
        element = np.random.choice(range(vocab_size),
                                   p=out)  # Sample character from the network according to the generated output probabilities
        gen_str.append(element)

        out = net.run_step(convert_to_one_hot([element], vocab_size), False)
    return gen_str
