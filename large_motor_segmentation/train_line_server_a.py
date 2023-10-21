import os

if __name__ == "__main__":
    local_training = False

    while True:

        with open("cross_valid_motor_list-a.txt", 'r') as f:
            x = f.readlines()
            if len(x) == 0:
                cross_valid_motor_list = []
            else:
                cross_valid_motor_list = x[0].split(',')

        with open("/home/ies/fu/train_output/trained.txt", 'r') as f:
            x = f.readlines()
            if len(x) == 0:
                file_list = []
            else:
                file_list = x[0].split(',')

        executable_str = None
        valid_motors = None
        train_txt = None
        for txt_file_name in sorted(os.listdir('train_line_a')):
            if executable_str is not None:
                break

            if 'cross_valid' in txt_file_name:
                for cross_valid_motor in cross_valid_motor_list:
                    if txt_file_name + cross_valid_motor not in file_list:
                        file_list.append(txt_file_name + cross_valid_motor)

                        with open('train_line_a' + '/' + txt_file_name) as f:
                            executable_str = f.read()

                        valid_motors = cross_valid_motor
                        train_txt = 'train_line_a' + '/' + txt_file_name

                        break
            else:
                if txt_file_name not in file_list:
                    file_list.append(txt_file_name)

                    with open('train_line_a' + '/' + txt_file_name) as f:
                        executable_str = f.read()

                    train_txt = 'train_line_a' + '/' + txt_file_name

        if executable_str is None:
            break
        else:
            with open("/home/ies/fu/train_output/trained.txt", 'w') as f:
                for i, trained_file_name in enumerate(file_list):
                    if i != 0:
                        f.write(',')
                    f.write(trained_file_name)

            exec(executable_str)
            # try:
            #     exec(executable_str)
            # except Exception:
            #     pass
