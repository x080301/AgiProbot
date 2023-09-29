import os

if __name__ == "__main__":

    while True:
        with open("/data/users/fu/trained.txt", 'r') as f:
            x = f.readlines()
        if len(x) == 0:
            file_list = []
        else:
            file_list = x[0].split(',')

        executable_str = None
        for txt_file_name in sorted(os.listdir('train_line')):
            if txt_file_name not in file_list:
                file_list.append(txt_file_name)

                with open("/data/users/fu/trained.txt", 'w') as f:
                    for i, trained_file_name in enumerate(file_list):
                        if i != 0:
                            f.write(',')
                        f.write(trained_file_name)

                train_txt = 'train_line' + '/' + txt_file_name
                with open(train_txt) as f:
                    executable_str = f.read()
                break

        if executable_str is None:
            break
        else:
            try:
                exec(executable_str)
            except Exception:
                pass
