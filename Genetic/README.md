# AIAssignment

## Genetic Algorithm

The implementation of the genetic algorithm consists of six scripts:

### geneticAgentParallel.py

Run this to begin training the genetic algorithm. You can change the parameters at the bottom of the file.
The default is for the population of snakes to be 1000, the mutation rate 0.7 and the crossover rate 0.3.
This configuration should begin to reach the best snakes it can achieve at around 40 generations.

For each generation, the best performing snake by fitness will be saved in savedModels. The fitnesses rankings of each snake of each generation will be stored in fitnesses, while the max scores (ie how many apples the snake gets) of each snake of each generation will be stored in scores.
Every snake produced in a generation is saved at the end of that generation, in case the user needs to resume later.

The algorithm will run in parallel, and will therefore run faster depending on how many cores your processor has.

### geneticAgentParallelResume.py

Every generation in the genetic algorithm will be saved - use this to resume training where you left off after stopping the geneticAgentParellel.py script.

### getStats.py

This script will produce summary statistics of how well the snake performed in each generation. Run it after training the model.

### replayGenerations.py

This script will replay each of the best models from every generation, so you can see how the snake improves over time. You can set the range of 
generations to replay at the bottom of the script.

### iterateBestSnake.py

This script will play 100 iterations of a chosen snake and then produce summary statistics. You can specify which snake to play at the bottom of the script.

### snake_env_genetic.py

Very similar to the other iterations of snake_env, with the key difference being that it tracks the snakes performance according to a fitness
function that takes the number of apples a snake gets and the total time it survived for before dying. It also will kill the snake after 500 moves without
getting an apple, as otherwise we would get infinite looping snakes.


## Deep Reinforcement Learning

The implementation of the Deep Reinforcement Learning Algorithm consists of 3 scripts for the three differen implementations that are  
1. DQN  
2. DoubleDQN  
3. DQN with CNN  

And to accompany these three scripts are the 2 different game environments, the snake_env.py,  taken from  Harder, H. (2020). Snake Played by a Deep Reinforcement Learning Agent,  for the algorithms without a CNN model and a modified snake env snake_env_cnn for supporting agents that use a Cnn Model.

### DQN.py

Contains the script for the Standard DQN algorithm, that is an extension of the agents environment from Harder, H. (2020). Snake Played by a Deep Reinforcement Learning Agent, with modifications made to the learning model that has been used as well as the strucuture of the implementation to improve speed. 

### DoubleDQN.py

Contains the script for the Double DQN algorihtm, that extends the DQN agent with 2 different learning models, the target and an online model, to avoid the overestiamtion problem that the standard DQN faces. 

## DNNDQN.py

The contains the implementation of the standard DQN algorithm with a state space that has been replaced with the calculated image of the game using the coordinates of the sanke and the food repersented in a 2D array that has been implemented in the sanke_env_cnn.py

### Requirements.txt

This contains all the dependencies required to run the different scripts. 





