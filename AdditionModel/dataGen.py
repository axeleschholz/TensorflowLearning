#use this to create a new 
import numpy as np
import itertools
import random

equations = []
targets = []
combinations = list(itertools.combinations(range(1,100), 2))
random.shuffle(combinations)
for each in combinations:
    operator = random.randint(0,1)
    equations.append([each[0],operator,each[1]])
    result = (each[0]+each[1]) if (operator == 1) else (each[0]-each[1])
    targets.append([result])

print(len(equations))
    
test_eq = np.array(equations[0:1000])
test_targets = np.array(targets[0:1000])

training_eq = np.array(equations[1000:])
training_targets = np.array(targets[1000:])