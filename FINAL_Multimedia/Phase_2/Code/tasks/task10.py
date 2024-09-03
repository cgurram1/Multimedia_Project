import sys
import os
sys.path.append(os.path.join(os.getcwd(), "../../code"))
path = os.getcwd()
import tasks.task7 as task7

def task10():
    query_label = int(input("Enter Label: \n"))
    task7.task7(query_label)

if __name__ == "__main__":
    task10()