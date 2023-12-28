import os
import subprocess
import socket

if __name__ == '__main__':

    hostname = socket.gethostname()
    if 'iesservergpu' in hostname:
        training_txt = 'training_in_line_sever.txt'
    else:
        training_txt = 'training_in_line_workstation.txt'

    while True:
        with open(training_txt, 'r') as file:
            lines = file.readlines()

        cmd_to_save = []
        cmd_already_found = False
        end_flag = True
        for single_line in lines:
            single_line = single_line.strip('\n')

            if cmd_already_found:
                cmd_to_save.append(single_line + '\n')
            else:
                if single_line[0] == '#':
                    cmd_to_save.append(single_line + '\n')
                else:
                    cmd = single_line
                    cmd_already_found = True
                    end_flag = False
                    cmd_to_save.append('#' + single_line + '\n')
                    cmd_to_save.append('running\n')

        with open(training_txt, 'w') as file:
            file.writelines(cmd_to_save)

        if end_flag:
            break

        result = subprocess.run(cmd, shell=True, text=True, stdout=None,  # subprocess.PIPE,
                                stderr=subprocess.PIPE)

        new_training_in_line_txt = []
        belong_to_the_same_demo = False
        for single_line in cmd_to_save:
            if belong_to_the_same_demo:
                if '----' in single_line:
                    belong_to_the_same_demo = False
                    new_training_in_line_txt.append(single_line)
                else:
                    new_training_in_line_txt.append('#' + single_line)
            else:

                if 'running' in single_line:
                    if result.returncode != 0:
                        new_training_in_line_txt.append('#' + str(result.stderr).replace('\n', ' ') + '\n')
                        belong_to_the_same_demo = True
                else:
                    new_training_in_line_txt.append(single_line)

        with open(training_txt, 'w') as file:
            file.writelines(new_training_in_line_txt)

        # for single_line in lines:
        #     single_line = single_line  # .strip('\n')
        #     if single_line[0] == '#':
        #         cmd_to_save.append(single_line)
        #         if '----' in single_line:
        #             running_flag = True
        #     else:
        #         cmd_to_save.append('#' + single_line)
        #         if running_flag:
        #             result = subprocess.run(single_line, shell=True, text=True, stdout=None,  # subprocess.PIPE,
        #                                     stderr=subprocess.PIPE)
        #             end_flag = False
        #
        #             if result.returncode != 0:
        #                 cmd_to_save.append('#' + str(result.stderr).replace('\n', ' ') + '\n')
        #                 running_flag = False
        #
        # if end_flag:
        #     break
        #
        # print(cmd_to_save)
