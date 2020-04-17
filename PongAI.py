import argparse
import os

from pong import *

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras.optimizers import Adam

from rl.agents.dqn import DQNAgent
from rl.policy import BoltzmannQPolicy, EpsGreedyQPolicy, LinearAnnealedPolicy
from rl.memory import SequentialMemory

if __name__ == '__main__':
    # get the script directory
    curdir = os.path.dirname(os.path.abspath(__file__))

    # Construct the argument parser
    parser = argparse.ArgumentParser()

    # Switches
    parser.add_argument("-train", required=False, dest='train', action='store_const', const=True,
                        default=False, help="Train the data and save the model")

    # Parse the commandline
    args = parser.parse_args()

    # initialize the game class
    env = Pong()

    # number of possible actions; 3 for the Pong game
    nb_actions = env.action_space.n

    # Build the neural network model
    model = Sequential()
    model.add(Flatten(input_shape=(1,) + env.observation_space.shape))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(nb_actions, activation='linear'))
    print(model.summary())

    # Keep the previous actions and steps
    memory = SequentialMemory(limit=100000, window_length=1)

    # epsilon-greedy policy
    policy = LinearAnnealedPolicy(EpsGreedyQPolicy(), attr='eps', value_max=1, value_min=0.05, value_test=.05, nb_steps=20000)

    # Use Deep Q Network to train the agent
    dqn = DQNAgent(model=model, nb_actions=nb_actions, memory=memory, nb_steps_warmup=10, target_model_update=1e-2,
                    policy=policy, dueling_type='max')

    # compile the network
    dqn.compile(Adam(lr=1e-3), metrics=['mae'])

    # running in train mode
    if args.train:
        # fit the model for 100k steps
        history=dqn.fit(env, nb_steps=100000, visualize=False, verbose=1, action_repetition=2, log_interval=100000)

        # save the model
        dqn.save_weights('dqn_pong_weights.h5f', overwrite=True)

    else:
        # load the model
        dqn.load_weights(filepath='dqn_pong_weights.h5f')

        # play game for 10 rounds
        dqn.test(env, nb_episodes=10, visualize=True)
