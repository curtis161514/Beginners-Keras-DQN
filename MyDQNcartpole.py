from keras.layers import Dense, Activation
from keras.models import Sequential
from keras.optimizers import Adam
import numpy as np
import gym
import tensorflow as tf


env = gym.make('CartPole-v1')
state_size = env.observation_space.shape[0]
n_actions = env.action_space.n

#initalize replay buffer
capacity = 2000


buffer_s = np.zeros((capacity, state_size), dtype = np.float32)
buffer_a = np.zeros((capacity), dtype = np.int)
buffer_r = np.zeros(capacity)
buffer_s_ = np.zeros((capacity, state_size), dtype = np.float32)
buffer_done = np.zeros(capacity)

####################################################################################3
#------------------------build Q------------------------------------------------------
#######################################################################################
fc1_dims = 32
fc2_dims = 32
lr = 0.001

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
#---------------------For Eposode 1, M do------------------------------------------
###################################################################################

# input variables
num_episodes = 500
epsilon = 1
epsilon_decay = .996
min_epsilon = 0.01
batch_size = 32
gamma = 0.95


mem_counter = 0
sample = np.array([], dtype = int)
epsilon_hist = []
action_space = [i for i in range(n_actions)]


for e in range(num_episodes):
    print(e)
    done = False
    score = 0
    state = env.reset()
    
    
    #for t seconds
    for t in range (200):
        
        #with prob epsilon select random action from action_space
        rand = np.random.random()
        if rand < epsilon:
            action = np.random.choice(action_space)
        else:
            action = np.argmax(Q.predict(np.asarray([state])))
            
        #advance Environment with max action
        state_, reward, done, info = env.step(action)
        
        reward = reward if not done else -10
            
        #store transition
        index = mem_counter % capacity
        buffer_s[index] = state
        buffer_a[index] = action
        buffer_r[index] = reward
        buffer_s_[index] = state_
        buffer_done[index] = 1 - int(done)
        
        
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
        #print(Q_now)


        #bellman update
        Y= Q_now.copy()
        
        if mem_counter < batch_size:
            num_samples = mem_counter
        else: 
            num_samples = batch_size
        
        for q in range(num_samples):
            Y[q,sample_a[q]] = sample_r[q] + gamma *max(Q_next[q])*buffer_done[q]
            
        #fit Q with samples
        Q.fit(sample_s, Y, epochs = 1,verbose = 0)

        #advance state to state_
        state = state_
        
         
        #decrement epsilon
        if epsilon > min_epsilon:
            epsilon *=epsilon_decay
        else:
            epsilon = min_epsilon
        
        #fixed target update
        if mem_counter % 20 == 0:
            Q_Target = Q
        
        #get the score for the episode
        #score += reward
        if done:
            break
        

    print('score ', t, ' e ', epsilon)



