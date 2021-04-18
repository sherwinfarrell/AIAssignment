from snake_env import Snake

import random
import numpy as np
from keras import Sequential
from collections import deque
from keras.layers import Dense
import matplotlib.pyplot as plt
from keras.optimizers import Adam
from plot_script import plot_result
import time




if __name__ == '__main__':

    name = None

    #Starting Episolon as one, so that no moves will be predicted till training happens with the neural network. 
    epsilon = 1


    #The value episilon will be reduced by
    gamma = .95

    #Sample size that will be used to train the neural network memory to avoid overfitting
    batch_size = 128

    #The minimum epsilon value can go to. 0.01 shows that only 0.01 of the times will the move be a random move,
    # The rest of times the move will be predicted using the neural net
    epsilon_min = .01
    #To reduce the epsilon value every time replay happens
    epsilon_decay = .995
    lr = 0.00025

    # Build Neural Network For Learning
    memory = deque(maxlen=3000)

    model = Sequential()
    model.add(Dense(248, input_shape=(12,), activation='relu'))

    model.add(Dense(128, activation='relu'))
    model.add(Dense(248, activation='relu'))
    model.add(Dense(128, activation='relu'))

    model.add(Dense(4, activation='softmax'))
    model.compile(loss='mse', optimizer=Adam(lr=lr))

    # Number of episodes that of the game. Each game will be played till the game is done.
    episodes = 20

    #The game environment that will be used to make the moves. 
    env = Snake()

    # To Keep track of the scores. 
    scores = []

    # Start of epsidoes loop. 
    for e in range(episodes):

        #Reset the env and get the state space. 
        # State space 
        state = env.reset()

        #Reshape the state to prepare for prediction. Current shape is 12, after reshaping it will be 1,12 to prepare it to be predicted.
        # Neural Net requires shape (None, 12), where None is the placeholder for the input size. 
        state = np.reshape(state, (1, env.state_space))
        score = 0
        max_steps = 10000
        for i in range(max_steps):
            action = None

            # Episolon greedy Action prediction
            if np.random.rand() <= epsilon:
                action = random.randrange(4)
            else:
                act_values = model.predict(state)
                #Select best action based on the qvalue. 
                action =  np.argmax(act_values[0])

            
            prev_state = state
            #Get Next state and save to memory to be used in replay memory for state action next state calculations. 
            next_state, reward, done, _ = env.step(action)
            score += reward
            next_state = np.reshape(next_state, (1, env.state_space))
            memory.append((state, action, reward, next_state, done))
            state = next_state

            # Replay happens here. 
            if batch_size > 1:
                # Only if Mem is greater than batch size start the replay. 
                if len(memory) < batch_size:
                    print("doning nothing")
                else:
                    # Extract all data points from memory to be used for evaluate the q values. 
                    minibatch = random.sample(memory, batch_size)
                    states = np.array([i[0] for i in minibatch])
                    actions = np.array([i[1] for i in minibatch])
                    rewards = np.array([i[2] for i in minibatch])
                    next_states = np.array([i[3] for i in minibatch])
                    dones = np.array([i[4] for i in minibatch])


                    # Reduce the dimension of the shape from batch_size, 1, 12 to Batch_size, 12 as it was expanded earlier only for batch size of 1

                    states = np.squeeze(states)
                    next_states = np.squeeze(next_states)

                    #Predict the q_values of the next states 
                    next_states_qval = model.predict_on_batch(next_states)

                    #Use those q_values to calculate the q_values of the current states and actions. Bellman Equation. 
                    new_q_values = rewards + gamma*(np.amax(next_states_qval, axis=1))*(1-dones)

                    # Predict the Q_values, which will be adjusted for the current action using the newly calculated q_values.
                    predicte_q_values = model.predict_on_batch(states)

                    ind = np.array([i for i in range(batch_size)])

                    # Update the all the predicted values for the current action given the state to the newly calucated value.
                    predicte_q_values[[ind], [actions]] = new_q_values

                    #Fit the neural network with that. 
                    model.fit(states, predicte_q_values, epochs=1, verbose=0)

                    #Reduce the Epislon using the epsilon decay factor.
                    if epsilon > epsilon_min:
                        epsilon *= epsilon_decay
            # If the game is done, by loosing mostly then break and print results. 
            if done:
                print(f'final state before dying: {str(prev_state)}')
                print(f'episodes: {e+1}/{episodes}, score: {score}')
                break
        scores.append(score)
    
    plt.plot(range(1,episodes+1),scores)
    plt.xlabel("episodess", fontsize = 14)
    plt.ylabel("Scores", fontsize = 14)
    plt.title("Scores For Every episodes")
    plt.show()
    
    