import os

if __name__ == '__main__':

    while True:
        with open('training_in_line.txt', 'r') as file:
            lines = file.readlines()

        cmd_to_save = []
        running_flag = True
        for single_line in lines:
            single_line = single_line.strip('\n')
            if single_line[0] == '#':
                cmd_to_save.append(single_line)
                if '----' in single_line:
                    running_flag = True
            else:
                if running_flag:
                    
                    os.system(single_line)
                cmd_to_save.append('#' + single_line)

        print(cmd_to_save)
        break
