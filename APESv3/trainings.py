import os

if __name__ == '__main__':

    while True:
        with open('training_in_line.txt', 'r') as file:
            lines = file.readlines()

        for single_line in lines:
            print(single_line)

        break
