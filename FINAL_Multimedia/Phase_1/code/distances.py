import math
def euclidean(inputVector, databaseVector):
    return(math.sqrt(sum((x - y) ** 2 for x, y in zip(inputVector, databaseVector))))