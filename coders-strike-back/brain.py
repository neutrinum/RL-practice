# -*- coding: utf-8 -*-
"""
Created on Wed Feb 14 13:48:20 2018

@author: Adrià Mompó Alepuz

Double Deep Q-Network implementation in TensorFlow for solving a minigame
called 'Coders Strike Back', from the website CodinGame
"""

import tensorflow as tf
import numpy as np
import environment as env
import random
import time

import matplotlib.pyplot as plt

from argparse import ArgumentParser

# these are the parameters that have produced the best results so far
epsilon = 1
epsilon_ = 0.2
epsilon_steps = 1600000
alpha = 0.5 
gamma = 0.95
learning_rate = 7e-8 
lr_start = 100000 
lr_steps = 1000000 
lr_final = 3e-9
num_episodes = 16000  
batch_size = 128
memo_min = 1024 * 8
memo_max = 1024 * 16
tau = 0.001

print_metrics = True
on_policy = False

animations = True
load_model = False
save_model = True
path_model = 'dqn/'

def build_parser():
    parser = ArgumentParser()
    parser.add_argument('--epsilon-init', type=float,
            dest='epsilon_init', help='initial value for epsilon in the epsilon greedy policy (default %(default)s)',
            metavar='EPSILON', default=epsilon)
    parser.add_argument('--epsilon-final', type=float,
            dest='epsilon_final', help='final value for epsilon in the epsilon greedy policy (default %(default)s)',
            metavar='EPSILON', default=epsilon_)
    parser.add_argument('--epsilon-steps', type=int,
            dest='epsilon_steps', help='number of steps from epsilon init to final (default %(default)s)',
            metavar='EPSILON', default=epsilon_steps)
    parser.add_argument('--alpha', type=float,
            dest='alpha', help='alpha value for DQN algorithm (default %(default)s)',
            metavar='ALPHA', default=alpha)
    parser.add_argument('--gamma', type=float,
            dest='gamma', help='gamma value for DQN algorithm (default %(default)s)',
            metavar='GAMMA', default=gamma)
    parser.add_argument('--learning_rate', type=float,
            dest='learning_rate', help='learning rate for backprop (default %(default)s)',
            metavar='LEARNING', default=learning_rate)
    parser.add_argument('--num-episodes', type=int,
            dest='num_episodes', help='iterations or time steps (default %(default)s)',
            metavar='EPISODES', default=num_episodes)
    parser.add_argument('--batch-size', type=int,
            dest='batch_size', help='batch size for backprop (default %(default)s)',
            metavar='LEARNING', default=batch_size)
    parser.add_argument('--memory-size', type=int,
            dest='memo_max', help='max memory events saved (default %(default)s)',
            metavar='MEMORY', default=memo_max)
    parser.add_argument('--memory-min', type=int,
            dest='memo_min', help='min memory events to start training (default %(default)s)',
            metavar='MEMORY', default=memo_min)
    parser.add_argument('--print-cost', type=bool,
            dest='print_metrics', help='max amount of memory events saved (default %(default)s)',
            metavar='PRINT', default=print_metrics)
    parser.add_argument('--on-policy', type=bool,
            dest='on_policy', help='wether to use a pre-defined policy with epsilon probability instead of random (default %(default)s (random policy))',
            metavar='POLICY', default=on_policy)

    return parser

def load_parser(options):
    global epsilon, epsilon_, epsilon_steps, alpha, gamma, learning_rate, num_episodes, batch_size, memo_min, memo_max, print_metrics, on_policy
    epsilon = options.epsilon_init
    epsilon_ = options.epsilon_final
    alpha = options.alpha 
    gamma = options.gamma
    learning_rate = options.learning_rate 
    num_episodes = options.num_episodes 
    batch_size = options.batch_size
    memo_min = options.memo_min
    memo_max = options.memo_max
    print_metrics = options.print_metrics
    on_policy = options.on_policy


def create_placeholders(n_x, n_y):
    """
    Creates the placeholders for the tensorflow session.
    
    Arguments:
    n_x -- scalar, dim of input (state_dim)
    n_y -- scalar, number of classes (action_dim)
        
    Returns:
    X -- placeholder for the data input, of shape [n_x, None] and dtype "float"
    Y -- placeholder for the input labels, of shape [n_y, None] and dtype "float"
    """

    ### START CODE HERE ### (≈2 lines)
    X = tf.placeholder('float',[None, n_x])
    Y = tf.placeholder('float',[None, n_y])
    ### END CODE HERE ###
    
    return X, Y
    
    
def initialize_parameters():
    """
    Initializes weight parameters to build a neural network with tensorflow. The shapes are:
                        W1 : [4, 4, 3, 8]
                        W2 : [2, 2, 8, 16]
    Returns:
    parameters -- a dictionary of tensors containing W1, W2
    """
    
    tf.set_random_seed(1)
    # shape = (in,out)
    W1 = tf.get_variable("W1on", [state_dim,32], initializer = tf.contrib.layers.xavier_initializer(seed=1))
    b1 = tf.get_variable("b1on", [1,32], initializer = tf.zeros_initializer())
    W2 = tf.get_variable("W2on", [32,32], initializer = tf.contrib.layers.xavier_initializer(seed=1))
    b2 = tf.get_variable("b2on", [1,32], initializer = tf.zeros_initializer())
    W3 = tf.get_variable("W3on", [32,action_dim], initializer = tf.contrib.layers.xavier_initializer(seed=1))
    b3 = tf.get_variable("b3on", [1,action_dim], initializer = tf.zeros_initializer())
    
    W1_tg = tf.get_variable("W1tg", [state_dim,32], initializer = tf.contrib.layers.xavier_initializer(seed=1))
    b1_tg = tf.get_variable("b1tg", [1,32], initializer = tf.zeros_initializer())
    W2_tg = tf.get_variable("W2tg", [32,32], initializer = tf.contrib.layers.xavier_initializer(seed=1))
    b2_tg = tf.get_variable("b2tg", [1,32], initializer = tf.zeros_initializer())
    W3_tg = tf.get_variable("W3tg", [32,action_dim], initializer = tf.contrib.layers.xavier_initializer(seed=1))
    b3_tg = tf.get_variable("b3tg", [1,action_dim], initializer = tf.zeros_initializer())
    
    # consider implementing clipping

    param_online = {"W1": W1,
                    "b1": b1,
                    "W2": W2,
                    "b2": b2,
                    "W3": W3,
                    "b3": b3}
    
    param_target = {"W1": W1_tg,
                    "b1": b1_tg,
                    "W2": W2_tg,
                    "b2": b2_tg,
                    "W3": W3_tg,
                    "b3": b3_tg}
    
    return param_online, param_target


def forward_propagation(X, param_online, param_target):
    """
    Implements vanilla neural network, 3 layers
    
    Arguments:
    X -- input dataset placeholder, of shape (input size, number of examples)
    parameters -- python dictionary containing your parameters "W1", "W2"
                  the shapes are given in initialize_parameters

    Returns:
    Z3 -- the output of the last LINEAR unit
    """
    
    W1 = param_online['W1']
    b1 = param_online['b1']
    W2 = param_online['W2']
    b2 = param_online['b2']
    W3 = param_online['W3']
    b3 = param_online['b3']
    
    #X = tf.nn.l2_normalize(X, dim=0) #try changing dim (dim0 = training example, dim1 = features)
    
    Z1 = tf.add(tf.matmul(X,W1),b1)             # Z1 = np.dot(W1, X) + b1
    #Z1 = tf.nn.l2_normalize(Z1, dim=0) #try changing dim
    Z1 = tf.nn.dropout(Z1,0.7)
    A1 = tf.nn.softplus(Z1)                         # A1 = relu(Z1)
    Z2 = tf.add(tf.matmul(A1,W2),b2)            # Z2 = np.dot(W2, a1) + b2
    #Z2 = tf.nn.l2_normalize(Z2, dim=0) #try changing dim
    Z2 = tf.nn.dropout(Z2,0.7)
    A2 = tf.nn.softplus(Z2)                         # A2 = relu(Z2)
    Z3 = tf.add(tf.matmul(A2,W3),b3)  
    
    
    W1_tg = param_target['W1']
    b1_tg = param_target['b1']
    W2_tg = param_target['W2']
    b2_tg = param_target['b2']
    W3_tg = param_target['W3']
    b3_tg = param_target['b3']
    
    Z1_tg = tf.add(tf.matmul(X,W1_tg),b1_tg)             # Z1 = np.dot(W1, X) + b1
    #Z1_tg = tf.nn.l2_normalize(Z1_tg, dim=0) #try changing dim
    Z1_tg = tf.nn.dropout(Z1_tg, 0.7)
    A1_tg = tf.nn.softplus(Z1_tg)                         # A1 = relu(Z1)
    Z2_tg = tf.add(tf.matmul(A1_tg,W2_tg),b2_tg)            # Z2 = np.dot(W2, a1) + b2
    #Z2_tg = tf.nn.l2_normalize(Z2_tg, dim=0) #try changing dim
    Z2_tg = tf.nn.dropout(Z2_tg, 0.7)
    A2_tg = tf.nn.softplus(Z2_tg)                         # A2 = relu(Z2)
    Z3_tg = tf.add(tf.matmul(A2_tg,W3_tg),b3_tg) 

    print(Z3.shape)
    return Z3, Z3_tg


def compute_cost(Z3, Y):
    """
    Computes the cost
    
    Arguments:
    Z3 -- output of forward propagation (output of the last LINEAR unit), of shape (6, number of examples)
    Y -- "true" labels vector placeholder, same shape as Z3
    
    Returns:
    cost - Tensor of the cost function
    """
    
    cost = tf.reduce_mean(tf.square(Z3 - Y))
    
    #regularizers = tf.nn.l2_loss(weights_1) + tf.nn.l2_loss(weights_2) + \
    #               tf.nn.l2_loss(weights_3) + tf.nn.l2_loss(weights_4) + \
    #               tf.nn.l2_loss(weights_5) + tf.nn.l2_loss(weights_6)
    #cost = tf.reduce_mean(cost + beta * regularizers)
    
    return cost


def update_target_weights(param_online, param_target):
    """ 
    Updates the parameters of the target neural network with those of the online NN
    """
    target_update_op = []
    
    for var in param_target:
        target_update_op.append(param_target[var].assign(param_online[var].value() * tau
                                + param_target[var].value() * (1-tau)))
    #W1 = param_online['W1']
    #W1_tg = param_target['W1']
    #W1_tg.assign(W1.value() * tau + W1_tg.value() * (1-tau)).op.run()
    
    return target_update_op
    

metrics = plt.figure(1)

def plotter(costs,rewards,values,final=False):
    global metrics
    
    metrics.clf()
    
    time_now = time.clock() - time_start
                
    m1 = metrics.add_subplot(311)                
    m1.plot(np.squeeze(costs))
    m1.set_ylabel('cost')
    #m1.xlabel('episodes')
    m1.set_title('epsilon %.5f'%epsilon + ' -- ' + 
              str(int(time_now//3600)) + ':' + str(int((time_now%3600)//60)) + ':' +
              str(int((time_now%3600)%60)) + ' -- ' +
              str(total_steps) + ' steps')
    
    m2 = metrics.add_subplot(312)                
    m2.plot(np.squeeze(rewards))
    m2.set_ylabel('reward')
    #m2.xlabel('episodes')
    #plt.title("Learning rate =" + str(learning_rate))
    
    m3 = metrics.add_subplot(313)                
    m3.plot(np.squeeze(values))
    m3.set_ylabel('value')
    m3.set_xlabel('episodes')
    #plt.title("Learning rate =" + str(learning_rate))
    
    if not final:
        plt.pause(0.05)
    else:
        plt.show()
    

state_dim = 6
action_dim = 35

total_steps = 0
time_start = None

def main():
    global epsilon, time_start, total_steps, learning_rate
    
    # Initialize lists
    costs = []
    rewards = []
    values = []
    
    memory = []

    # Parse arguments
    parser = build_parser()
    options = parser.parse_args()
    load_parser(options)
    # epsilon and learning rate decrement 
    epsilon_decr = (epsilon - epsilon_) / epsilon_steps
    lr_decr = (learning_rate - lr_final) / lr_steps       
    
    tf.set_random_seed(1)
    seed = 0

    # Initialize TF variables and graph
    X, Y = create_placeholders(state_dim, action_dim) # (state_dim, action_dim)
    
    param_online, param_target = initialize_parameters()
    
    onlineNN, targetNN = forward_propagation(X, param_online, param_target)
    
    cost = compute_cost(onlineNN, Y)
    
    # Target network updater operation
    target_update_op = update_target_weights(param_online, param_target)
    
    # Backpropagation
    # optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(cost)
    # With clipping gradients: https://stackoverflow.com/a/43486487/5731130
    optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate)
    gradients, variables = zip(*optimizer.compute_gradients(cost))
    gradients, _ = tf.clip_by_global_norm(gradients, 1.0)
    optimize = optimizer.apply_gradients(zip(gradients, variables))
    # visualize tf.global_norm(gradients) to see the usual range, then choose the clipping threshold (initial guess 2.0)
    
    init = tf.global_variables_initializer()
    saver = tf.train.Saver()
    
    ### MAIN PROGRAM
    config = tf.ConfigProto(device_count = {'GPU': 0}) # seems it is faster with CPU
    with tf.Session(config=config) as sess:
        
        # Run the initialization
        if load_model:
            print('Restoring model ...')
            ckpt = tf.train.get_checkpoint_state('dqn/')
            saver.restore(sess,ckpt.model_checkpoint_path)
        else:
            sess.run(init)
        
        # Initialize environment and clock
        env.setup()
        time_start = time.clock()
        
        # Perform the training loop
        for e in range(num_episodes):
            
            # Setup new episode (initialize environment)
            env.setup()
            # Initial state
            s = env.get_state()
            
            t = 0
            rewEp = 0
            valueEp = 0
            costEp = 0
            
            if e%50 == 0:
                print(e,' episode starting')
            
            while not env.race_finished(): # initialization of environment automatically when this method returns true
                #print(t)
                ### SEQUENCE / SIMULATION STEP --- START ###
                # Create current sequence
                seq = [s]
                
                # Simulate and obtain info from the environment
                Q = onlineNN.eval(feed_dict={X:[s]})
                
                # Pick action, epsilon-greedy
                if np.random.random() > epsilon: # greedy policy
                    a = np.argmax(Q)
                elif on_policy: # hard-coded policy
                    a = env.get_next_action()
                else: # random policy
                    a = np.random.randint(action_dim)
                
                # Get reward and new state after performing action 'a', updating simulation
                r,s = env.action(a)
                
                # Save rest of sequence
                seq.extend([a,r,s])
                
                if len(memory) < memo_max:
                    memory.append(seq)
                else:
                    memory.pop(0)
                    memory.append(seq)
                ### SEQUENCE / SIMULATION STEP --- END ###
                
                """ LEARN ON NEURAL NETWORK WITH DDQN """
                # When there's enough memory and every 4 iterations
                if len(memory) >= memo_min and t % 4 == 0: # set min_memo proportional to batch_size

                    # Creation of minibatch used for learning
                    replays = np.array(random.sample(memory, batch_size))
                    
                    state = np.vstack(np.array(replays[:,0])) # shape batch_size, state_dim
                    action = np.vstack(np.array(replays[:,1]))
                    reward = np.vstack(np.array(replays[:,2]))
                    state_ = np.vstack(np.array(replays[:,3]))
                    
                    """ DOUBLE DQN ALGORITHM """
                    # calculate y_DoubleQ (y_Q)
                    a_max = np.argmax(onlineNN.eval(feed_dict={X:state_}), axis=1)

                    # y_Q = reward + gamma * Q(S',target_network)[a_max_Q(S',online_network)]
                    y_Q = reward.squeeze() + gamma * targetNN.eval(feed_dict={X:state_})[np.arange(batch_size), a_max]
                    
                    # This is the evaluation of current online neural network of the state (for all actions)
                    Q_eval = onlineNN.eval(feed_dict={X:state})
                    # Value to which the network is going to aim at its output for the action taken
                    Q_update = y_Q - Q_eval[np.arange(batch_size),action.squeeze()]
                    # The full vector of outputs with the the index of the action taken updated
                    Q_eval[np.arange(batch_size),action.squeeze()] += alpha * Q_update
                          
                    # Create the minibatch used for learning
                    batch_X = state
                    batch_Y = Q_eval
                    
                    """ UPDATE ONLINE NEURAL NETWORK """
                    # Train on minibatch generated
                    _ , batch_cost = sess.run([optimize,cost],feed_dict={X:batch_X, Y:batch_Y})
                    
                    costEp += batch_cost / batch_size
                
                    """ UPDATE TARGET NEURAL NETWORK """
                    sess.run(target_update_op)
                    
                # END OF TIMESTEP
                t += 1
                total_steps += 1
                rewEp += r
                valueEp += np.mean(Q)
                
                # Hyperparameters update
                if len(memory) >= memo_min and epsilon > epsilon_:
                    epsilon -= epsilon_decr
                if total_steps > lr_start and learning_rate > lr_final:
                    learning_rate -= lr_decr
                
                # Print progress
                if e%50 == 0 and t%10 == 0:
                    print('=',end='',flush=True)
            
            ### END OF EPISODE (next e in for loop after this ident finishes)
            if e%50 == 0:
                print('> '+str(t)+' steps')
                print('%i timesteps in episode %i' % (t,e))
            
            costs.append(costEp/t*4)
            rewards.append(rewEp / t)
            values.append(valueEp / t)
            
            np.random.seed(seed)
            seed +=1
            
            # Show metrics every few episodes
            if print_metrics and e % 100 == 0:
                #print ("Cost after episode %i: %f" % (e, costs[e]))
                print ("Reward after episode %i: %f" % (e, rewards[e]))
                print ("Value after episode %i: %f" % (e, values[e]))
                #print ("-- memory length ", len(memory))
                # And plots
                plotter(costs,rewards,values)

            if animations and e % 1000 == 0:
                acts = np.array(memory)[len(memory)-t:len(memory)-1,1]
                env.save_animation(acts.tolist(),e)
                
            # Save model
            if save_model and e > 0 and e % 2000 == 0:
                print('Saving model ...')
                saver.save(sess, path_model+'model'+str(e)+'.ckpt')
        
        # END OF PROGRAM
        # plot metrics, save model
        if save_model:
            print('Saving model ...')
            saver.save(sess, path_model+'model'+str(e)+'.ckpt')
        if animations:    
            acts = np.array(memory)[len(memory)-t:len(memory)-1,1]
            env.save_animation(acts.tolist())
            
        plotter(costs,rewards,values,final=True)

if __name__ == '__main__':
    main()

