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




