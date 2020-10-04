from keras.layers import Dense, Activation
from keras.models import Sequential
from keras.optimizers import Adam
import numpy as np
import gym
import tensorflow as tf
import matplotlib as plt


#env = gym.make('CartPole-v1')
env = gym.make('LunarLander-v2')
state_size = env.observation_space.shape[0]
n_actions = env.action_space.n

#initalize replay buffer
capacity = 50000


buffer_s = np.zeros((capacity, state_size), dtype = np.float32)
buffer_a = np.zeros((capacity), dtype = np.int)
buffer_r = np.zeros(capacity)
buffer_s_ = np.zeros((capacity, state_size), dtype = np.float32)
buffer_done = np.zeros(capacity)

####################################################################################3
#------------------------build Q------------------------------------------------------
#######################################################################################
fc1_dims = 128
fc2_dims = 128
lr = 0.0001

#clear previous model just in case
tf.keras.backend.clear_session()

Q = Sequential([
                Dense(fc1_dims, input_shape = (state_size, )),
                Activation('relu'),
                Dense(fc2_dims),
                Activation('relu'),
                Dense(n_actions),
                Activation('linear')])
                

Q.compile(optimizer = Adam(lr = lr), loss = 'mse')

#################################################################################
#-----------------------------initalize Q_target---------------------------------
#################################################################################

Q_Target = Q

###################################################################################
#-------------------DQN algorithim-----------------------------------------
###################################################################################

# input variables
num_episodes = 1000
epsilon = 1
epsilon_decay = .9996
min_epsilon = 0.01
batch_size = 64
gamma = 0.99


mem_counter = 0
sample = np.array([], dtype = int)
action_space = [i for i in range(n_actions)]
scores = np.asarray([])


for e in range(num_episodes):
    done = False
    score = 0
    state = env.reset()
    
    
    while not done:
    
        #with prob epsilon select random action from action_space
        rand = np.random.random()
        if rand < epsilon:
            action = np.random.choice(action_space)
        else:
            action = np.argmax(Q.predict(np.asarray([state])))
        
        #decrement epsilon
        if epsilon > min_epsilon:
            epsilon *=epsilon_decay
        else:
            epsilon = min_epsilon
            
        #advance Environment with max action
        state_, reward, done, info = env.step(action)
              
        #store transition
        index = mem_counter % capacity
        buffer_s[index] = state
        buffer_a[index] = action
        buffer_r[index] = reward
        buffer_s_[index] = state_
        buffer_done[index] = 1 - int(done)
        
        #advance state to state_
        state = state_
        
        #sampe random batch from transition
        sample = np.random.choice(min(mem_counter+1,capacity),min(mem_counter+1,batch_size),replace = False)
        sample = np.sort(sample)
        mem_counter +=1
        
        #store minibatch of transitions
        sample_s = buffer_s[sample]
        sample_a = buffer_a[sample]
        sample_r = buffer_r[sample]
        sample_s_ = buffer_s_[sample]
        sample_done = buffer_done[sample]
        
       
        #learning
        Q_now = Q.predict(sample_s)
        Q_next = Q_Target.predict(sample_s_)
        
        #fixed target update
        if mem_counter % 50 == 0:
            Q_Target = Q
            
        #if there is not enough samples in replay buffer than just use what we have
        #for the bellman update
        if mem_counter < batch_size:
            num_samples = mem_counter
        else: 
            num_samples = batch_size
        
        #bellman update
        Y= Q_now.copy()
        
        for q in range(num_samples):
            Y[q,sample_a[q]] = sample_r[q] + gamma *max(Q_next[q])*buffer_done[q]
            
        #fit Q with updated samples
        Q.fit(sample_s, Y, epochs = 1,verbose = 0)
  
        #get the score for the episode
        score += reward
        
    scores = np.append(scores,score)
    
    if e > 10:
        moving_average = np.average(scores[e-10:e])
    else: moving_average = 0
        
    if moving_average > 200:
        Q.save('LunarLanderDQN.h5')
        break
    
    print('episode_',e ,' score_', score, ' average_', moving_average)

plt.pyplot.plot(scores)

