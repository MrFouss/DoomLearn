#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import division
from __future__ import print_function
from vizdoom import *
import itertools as it
from random import sample, randint, random
from time import time, sleep
import numpy as np
import skimage.color, skimage.transform
import tensorflow as tf
from tqdm import trange
import time as tm
import os

# Q-learning settings
learning_rate = 0.00025
# learning_rate = 0.0001
discount_factor = 0.99
epochs = 20
learning_steps_per_epoch = 2000
replay_memory_size = 10000

# NN learning settings
batch_size = 64

# Training regime
test_episodes_per_epoch = 100

# Other parameters
frame_repeat = 12
resolution = (60, 80)
screen_channels = 3
episodes_to_watch = 10

save_model = True
load_model = False
skip_learning = False

# Configuration file path
config_file = "basic"
config_file_path = "scenarios/" + config_file + ".cfg"

# savefile paths
save_dir = "trainings/training_" + config_file + "_" + tm.asctime( tm.localtime(tm.time()) )
model_savefile = save_dir + "/model_" + config_file + ".ckpt"
tensorboard_savefile = save_dir
parameters_savefile = save_dir + "/parameters_" + config_file + ".txt"

# Save simulation parameters
def save_simulation_parameters():
    os.makedirs(os.path.dirname(parameters_savefile), exist_ok=True)
    with open(parameters_savefile, 'w') as file:
        file.write('learning rate=' + str(learning_rate) + '\n')
        file.write('discount factor=' + str(discount_factor) + '\n')
        file.write('epochs=' + str(epochs) + '\n')
        file.write('learning steps per epoch=' + str(learning_steps_per_epoch) + '\n')
        file.write('test episodes  per epoch=' + str(test_episodes_per_epoch) + '\n')
        file.write('frame repeat=' + str(frame_repeat) + '\n')
        file.write('resolution=' + str(resolution) + '\n')
        file.write('screen channels=' + str(screen_channels) + '\n')
        file.write('optimizer=' + str(optimizer.get_name()) + '\n')
        file.write('\n------------VIZDOOM CONFIG FILE------------\n\n')
        with open(config_file_path, 'r') as configFile:
            lines = configFile.readlines()
            file.writelines(lines)

# Converts and down-samples the input image
def preprocess(img):
    img = skimage.transform.resize(img, list(resolution) + [screen_channels])
    img = img.astype(np.float32)
    return img

class ReplayMemory:
    def __init__(self, capacity):
        state_shape = (capacity, resolution[0], resolution[1], screen_channels)
        self.s1 = np.zeros(state_shape, dtype=np.float32)
        self.s2 = np.zeros(state_shape, dtype=np.float32)
        self.a = np.zeros(capacity, dtype=np.int32)
        self.r = np.zeros(capacity, dtype=np.float32)
        self.isterminal = np.zeros(capacity, dtype=np.float32)

        self.capacity = capacity
        self.size = 0
        self.pos = 0

    def add_transition(self, s1, action, s2, isterminal, reward):
        self.s1[self.pos, :, :, :] = s1
        self.a[self.pos] = action
        if not isterminal:
            self.s2[self.pos, :, :, :] = s2
        self.isterminal[self.pos] = isterminal
        self.r[self.pos] = reward

        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def get_sample(self, sample_size):
        i = sample(range(0, self.size), sample_size)
        return self.s1[i], self.a[i], self.s2[i], self.isterminal[i], self.r[i]

def variable_summaries(tensor):
  """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
  with tf.name_scope('summaries'):
    mean = tf.reduce_mean(tensor)
    tf.summary.scalar('mean', mean)
    with tf.name_scope('stddev'):
      stddev = tf.sqrt(tf.reduce_mean(tf.square(tensor - mean)))
    tf.summary.scalar('stddev', stddev)
    tf.summary.scalar('max', tf.reduce_max(tensor))
    tf.summary.scalar('min', tf.reduce_min(tensor))
    tf.summary.histogram('histogram', tensor)


def create_network(session, available_actions_count):
    # Create the input variables
    s1_ = tf.placeholder(tf.float32, [None] + list(resolution) + [screen_channels], name="State")
    a_ = tf.placeholder(tf.int32, [None], name="Action")
    target_q_ = tf.placeholder(tf.float32, [None, available_actions_count], name="TargetQ")

    # Add 2 convolutional layers with ReLu activation
    # conv1 = tf.contrib.layers.convolution2d(s1_, num_outputs=32, kernel_size=[7, 7], 
    #                                         stride=[2, 2],
    #                                         activation_fn=tf.nn.relu,
    #                                         weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
    #                                         biases_initializer=tf.constant_initializer(0.1))
    # conv2 = tf.contrib.layers.convolution2d(conv1, num_outputs=64, kernel_size=[7, 7], 
    #                                         stride=[2, 2],
    #                                         activation_fn=tf.nn.relu,
    #                                         weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
    #                                         biases_initializer=tf.constant_initializer(0.1))
    # maxPool1 = tf.contrib.layers.max_pool2d(conv2, kernel_size=[3, 3], stride=[2, 2])
    # conv3 = tf.contrib.layers.convolution2d(maxPool1, num_outputs=128, kernel_size=[3, 3], 
    #                                         stride=[1, 1],
    #                                         activation_fn=tf.nn.relu,
    #                                         weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
    #                                         biases_initializer=tf.constant_initializer(0.1))
    # maxPool2 = tf.contrib.layers.max_pool2d(conv3, kernel_size=[3, 3], stride=[2, 2])
    # conv4 = tf.contrib.layers.convolution2d(maxPool2, num_outputs=192, kernel_size=[3, 3], 
    #                                         stride=[1, 1],
    #                                         activation_fn=tf.nn.relu,
    #                                         weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
    #                                         biases_initializer=tf.constant_initializer(0.1))
    # conv4_flat = tf.contrib.layers.flatten(conv4)
    # fc1 = tf.contrib.layers.fully_connected(conv4_flat, num_outputs=256,                  
    #                                         activation_fn=tf.nn.relu,
    #                                         weights_initializer=tf.contrib.layers.xavier_initializer(),
    #                                         biases_initializer=tf.constant_initializer(0.1))

    # q = tf.contrib.layers.fully_connected(fc1, num_outputs=available_actions_count, 
    #                                         activation_fn=None,
    #                                         weights_initializer=tf.contrib.layers.xavier_initializer(),
    #                                         biases_initializer=tf.constant_initializer(0.1))
    # best_a = tf.argmax(q, 1)

    conv1 = tf.contrib.layers.convolution2d(s1_, num_outputs=32, kernel_size=[8, 8], 
                                            stride=[4, 4],
                                            activation_fn=tf.nn.relu,
                                            weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                            biases_initializer=tf.constant_initializer(0.1))
    conv2 = tf.contrib.layers.convolution2d(conv1, num_outputs=64, kernel_size=[4, 4], 
                                            stride=[2, 2],
                                            activation_fn=tf.nn.relu,
                                            weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                            biases_initializer=tf.constant_initializer(0.1))
    conv3 = tf.contrib.layers.convolution2d(conv2, num_outputs=64, kernel_size=[3, 3], 
                                            stride=[1, 1],
                                            activation_fn=tf.nn.relu,
                                            weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                            biases_initializer=tf.constant_initializer(0.1))
    conv3_flat = tf.contrib.layers.flatten(conv3)
    fc1 = tf.contrib.layers.fully_connected(conv3_flat, num_outputs=512,                  
                                            activation_fn=tf.nn.relu,
                                            weights_initializer=tf.contrib.layers.xavier_initializer(),
                                            biases_initializer=tf.constant_initializer(0.1))

    q = tf.contrib.layers.fully_connected(fc1, num_outputs=available_actions_count, 
                                            activation_fn=None,
                                            weights_initializer=tf.contrib.layers.xavier_initializer(),
                                            biases_initializer=tf.constant_initializer(0.1))
    best_a = tf.argmax(q, 1)


    with tf.name_scope('loss'):
        loss = tf.losses.mean_squared_error(q, target_q_)
        tf.summary.scalar('value', loss)

    optimizer = tf.train.RMSPropOptimizer(learning_rate)

    # Update the parameters according to the computed gradient using RMSProp.
    train_step = optimizer.minimize(loss)

    def function_learn(s1, target_q):
        feed_dict = {s1_: s1, target_q_: target_q}
        l, _, m = session.run([loss, train_step, merged], feed_dict=feed_dict)
        train_writer.add_summary(m, train_steps_since_start)
        return l

    def function_get_q_values(state):
        return session.run(q, feed_dict={s1_: state})

    def function_get_best_action(state):
        return session.run(best_a, feed_dict={s1_: state})

    def function_simple_get_best_action(state):
        return function_get_best_action(state.reshape([1, resolution[0], resolution[1], screen_channels]))[0]

    return function_learn, function_get_q_values, function_simple_get_best_action, optimizer


def learn_from_memory():
    """ Learns from a single transition (making use of replay memory).
    s2 is ignored if s2_isterminal """

    # Get a random minibatch from the replay memory and learns from it.
    if memory.size > batch_size:
        s1, a, s2, isterminal, r = memory.get_sample(batch_size)

        q2 = np.max(get_q_values(s2), axis=1)
        target_q = get_q_values(s1)
        # target differs from q only for the selected action. The following means:
        # target_Q(s,a) = r + gamma * max Q(s2,_) if isterminal else r
        target_q[np.arange(target_q.shape[0]), a] = r + discount_factor * (1 - isterminal) * q2
        learn(s1, target_q)


def perform_learning_step(epoch):
    """ Makes an action according to eps-greedy policy, observes the result
    (next state, reward) and learns from the transition"""

    def exploration_rate(epoch):
        """# Define exploration rate change over time"""
        start_eps = 1.0
        end_eps = 0.1
        const_eps_epochs = 0.1 * epochs  # 10% of learning time
        eps_decay_epochs = 0.6 * epochs  # 60% of learning time

        if epoch < const_eps_epochs:
            return start_eps
        elif epoch < eps_decay_epochs:
            # Linear decay
            return start_eps - (epoch - const_eps_epochs) / \
                               (eps_decay_epochs - const_eps_epochs) * (start_eps - end_eps)
        else:
            return end_eps

    s1 = preprocess(game.get_state().screen_buffer)

    # With probability eps make a random action.
    eps = exploration_rate(epoch)

    explorationRateSummary.value[0].simple_value = eps
    train_writer.add_summary(explorationRateSummary, train_steps_since_start)

    if random() <= eps:
        a = randint(0, len(actions) - 1)
    else:
        # Choose the best action according to the network.
        a = get_best_action(s1)
    reward = game.make_action(actions[a], frame_repeat)

    isterminal = game.is_episode_finished()
    s2 = preprocess(game.get_state().screen_buffer) if not isterminal else None

    # Remember the transition that was just experienced.
    memory.add_transition(s1, a, s2, isterminal, reward)

    learn_from_memory()


# Creates and initializes ViZDoom environment.
def initialize_vizdoom(config_file_path):
    print("Initializing doom...")
    game = DoomGame()
    game.load_config(config_file_path)
    game.set_window_visible(True)
    game.init()
    print("Doom initialized.")
    return game


if __name__ == '__main__':
    # Create Doom instance
    game = initialize_vizdoom(config_file_path)


    # Action = which buttons are pressed
    n = game.get_available_buttons_size()
    actions = [list(a) for a in it.product([0, 1], repeat=n)]

    # Create replay memory which will store the transitions
    memory = ReplayMemory(capacity=replay_memory_size)

    session = tf.Session()
    learn, get_q_values, get_best_action, optimizer = create_network(session, len(actions))
    saver = tf.train.Saver()
    if load_model:
        print("Loading model from: ", model_savefile)
        saver.restore(session, model_savefile)
    else:
        init = tf.global_variables_initializer()
        session.run(init)
    print("Starting the training!")

    # Save parameters of training
    save_simulation_parameters()

    # score summaries
    trainScoreSummary = tf.Summary(value=[tf.Summary.Value(tag="score/training value", simple_value=0)])
    testScoreSummary = tf.Summary(value=[tf.Summary.Value(tag="score/test value", simple_value=0)])
    explorationRateSummary = tf.Summary(value=[tf.Summary.Value(tag="exploration rate/value", simple_value=0)])
    trainOrTestSummary = tf.Summary(value=[tf.Summary.Value(tag="process/train(1) or test(-1)", simple_value=0)])
    train_steps_since_start = 0

    # tensor board summary merge
    merged = tf.summary.merge_all()

    # file writer
    train_writer = tf.summary.FileWriter(tensorboard_savefile, session.graph)

    time_start = time()
    if not skip_learning:
        for epoch in range(epochs):
            print("\nEpoch %d\n-------" % (epoch + 1))
            train_episodes_finished = 0
            train_scores = []

            print("Training...")
            game.new_episode()
            for learning_step in trange(learning_steps_per_epoch, leave=False):
                perform_learning_step(epoch)
                if game.is_episode_finished():
                    score = game.get_total_reward()
                    train_scores.append(score)
                    game.new_episode()
                    train_episodes_finished += 1

                    trainScoreSummary.value[0].simple_value = score
                    train_writer.add_summary(trainScoreSummary, train_steps_since_start)
                
                trainOrTestSummary.value[0].simple_value = 1
                train_writer.add_summary(trainOrTestSummary, train_steps_since_start)
                train_steps_since_start += 1

            print("%d training episodes played." % train_episodes_finished)

            train_scores = np.array(train_scores)

            print("Results: mean: %.1f±%.1f," % (train_scores.mean(), train_scores.std()), \
                  "min: %.1f," % train_scores.min(), "max: %.1f," % train_scores.max())

            print("\nTesting...")
            test_episode = []
            test_scores = []
            for test_episode in trange(test_episodes_per_epoch, leave=False):
                game.new_episode()
                while not game.is_episode_finished():
                    state = preprocess(game.get_state().screen_buffer)
                    best_action_index = get_best_action(state)

                    game.make_action(actions[best_action_index], frame_repeat)
                    train_steps_since_start += 1
                    trainOrTestSummary.value[0].simple_value = -1
                    train_writer.add_summary(trainOrTestSummary, train_steps_since_start-1)
                r = game.get_total_reward()
                test_scores.append(r)

                testScoreSummary.value[0].simple_value = r
                train_writer.add_summary(testScoreSummary, train_steps_since_start-1)

            test_scores = np.array(test_scores)
            print("Results: mean: %.1f±%.1f," % (
                test_scores.mean(), test_scores.std()), "min: %.1f" % test_scores.min(),
                  "max: %.1f" % test_scores.max())

            print("Saving the network weigths to:", model_savefile)
            saver.save(session, model_savefile)

            print("Total elapsed time: %.2f minutes" % ((time() - time_start) / 60.0))

    game.close()
    print("======================================")
    print("Training finished. It's time to watch!")

    # Reinitialize the game with window visible
    game.set_window_visible(True)
    game.set_mode(Mode.ASYNC_PLAYER)
    game.init()

    for _ in range(episodes_to_watch):
        game.new_episode()
        while not game.is_episode_finished():
            state = preprocess(game.get_state().screen_buffer)
            best_action_index = get_best_action(state)

            # Instead of make_action(a, frame_repeat) in order to make the animation smooth
            game.set_action(actions[best_action_index])
            for _ in range(frame_repeat):
                game.advance_action()

        # Sleep between episodes
        sleep(1.0)
        score = game.get_total_reward()
        print("Total score: ", score)
