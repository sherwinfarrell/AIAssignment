import matplotlib.pyplot as plt
import csv
import os

#This script produces the average fitness based on the saved models

print(os.listdir("scores"))

scoreList = []
aveScores = []
maxScores = []
for i in range(1,len(os.listdir("scores"))):

    with open("scores/gen" + str(i)) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            for entry in row:
                if(float(entry) - 15 < 0):
                    scoreList.append(0)
                else:
                    scoreList.append(float(entry) - 15)

    aveScores.append(sum(scoreList)/ len(scoreList))
    maxScores.append(max(scoreList))
    scoreList = []

    

print(aveScores)
print(maxScores)



scoreList = []
aveFitness = []
maxFitness = []
for i in range(1,len(os.listdir("fitnesses"))):

    with open("fitnesses/gen" + str(i)) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            for entry in row:
                scoreList.append(float(entry))

    
    aveFitness.append(sum(scoreList)/ len(scoreList))
    maxFitness.append(max(scoreList))
    scoreList = []

print(aveFitness)
print(maxFitness)

plt.plot(aveFitness)
plt.title("Genetic Algorithm - Average Fitness Across Generations")
plt.xlabel("Fitness")
plt.ylabel("Generations")
plt.show()

