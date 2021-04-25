from snake_env import Snake

import random
import numpy as np
from keras import Sequential
from collections import deque
from keras.layers import Dense
import matplotlib.pyplot as plt
from keras.optimizers import Adam
import time
from datetime import datetime


def create_model():
    model = Sequential()
    model.add(Dense(248, input_shape=(12,), activation='relu'))

    model.add(Dense(128, activation='relu'))
    model.add(Dense(248, activation='relu'))
    model.add(Dense(128, activation='relu'))

    model.add(Dense(4, activation='softmax'))
    model.compile(loss='mse', optimizer=Adam(lr=lr))
    return model

if __name__ == '__main__':

    name = None
    epsilon = 1
    gamma = .95
    batch_size = 128
    epsilon_min = .01
    epsilon_decay = .995
    lr = 0.00025
    memory = deque(maxlen=3000)
    min_sample = 512
    # Two models will be trained for extra stability. 
    online_model = create_model()
    target_model = create_model()
    target_model.set_weights(online_model.get_weights())
    min_counter = 20
    counter=0


    episodes = 100 
    env = Snake()
    rewards_list = []
    scores = []

    datetime1 = datetime.now()
    datetime2 = None
    for e in range(episodes):
        state = env.reset()
        state = np.reshape(state, (1, env.state_space))
        score = 0
        current_score = 0
        current_reward = 0
        max_steps = 10000
        for i in range(max_steps):
            action = None
            if np.random.rand() <= epsilon:
                action = random.randrange(4)
            else:
                act_values = online_model.predict(state)
                action =  np.argmax(act_values[0])
            # print(action)
            prev_state = state
            next_state, reward, done,env_score, _ = env.step(action)
            if not done:
                current_score = env_score
                if current_score >= 40:
                    datetime2 = datetime.now()
            current_reward += reward
            next_state = np.reshape(next_state, (1, env.state_space))
            memory.append((state, action, reward, next_state, done))
            state = next_state
            if batch_size > 1:
                if len(memory) < batch_size:
                    print("doning nothing")
                else:
                    minibatch = random.sample(memory, batch_size)
                    states = np.array([i[0] for i in minibatch])
                    actions = np.array([i[1] for i in minibatch])
                    rewards = np.array([i[2] for i in minibatch])
                    next_states = np.array([i[3] for i in minibatch])
                    dones = np.array([i[4] for i in minibatch])
                    ind = np.array([i for i in range(batch_size)])


                    states = np.squeeze(states)
                    next_states = np.squeeze(next_states)

                    q_next_target = target_model.predict_on_batch(next_states)
                    q_next_online = online_model.predict_on_batch(next_states)
                    q_current_online = online_model.predict_on_batch(states)

                    # Get the max action from the online model
                    max_actions = np.amax(q_next_online, axis=1)
                    q_target = q_current_online

                    # Q(S,A) = rewards + gamma * Q(S*, max)action(Q(S*,a)))*(1 - dones)
                    q_target[[ind], [actions]] = rewards + gamma * q_next_target[[ind],[max_actions.astype(int)]]*(1-dones)

                    

                    online_model.fit(states, q_target, epochs=1, verbose=0)

                    if done:
                            counter += 1

                    if counter > min_counter:
                        target_model.set_weights(online_model.get_weights())
                        counter = 0
                    if epsilon > epsilon_min:
                        epsilon *= epsilon_decay
            if done:
                print(f'final state before dying: {str(prev_state)}')
                print(f'episodes: {e+1}/{episodes}, reward: {current_reward}')
                print(f'episodes: {e+1}/{episodes}, score: {current_score}')
                break
        rewards_list.append(current_reward)
        scores.append(current_score)
    
  

    print("The max reward in all the episodes is "+ str(env.maximum))
    if datetime2:
        difference = datetime2 - datetime1
        print(f"The time to first 50 score is : {difference}")
    
        # print(scores)
    print("The mean score over all the epsiode is: ", sum(scores)/episodes)
    print("The average reward over all the episodes is: ", sum(rewards_list)/episodes)

    
    plot1 = plt.figure(1)
    plt.plot(range(1,episodes+1),scores)
    plt.xlabel("Episodess", fontsize = 14)
    plt.ylabel("Scores", fontsize = 14)
    plt.title("Scores For Every Episode")
    # plt.show()

    plot1 = plt.figure(2)
    plt.plot(range(1,episodes+1),rewards_list)
    plt.xlabel("episodess", fontsize = 14)
    plt.ylabel("Rewards", fontsize = 14)
    plt.title("Rewards For Every episodes")
    plt.show()

    
    