import os

file_list = []

keep_training = True
while keep_training:
    keep_training = False

    for txt_file_name in os.listdir('train_line'):
        if txt_file_name not in file_list:
            file_list.append(txt_file_name)
            keep_training = True
            with open('train_line' + '/' + txt_file_name) as f:
                executable_str = f.read()
            try:
                exec(executable_str)
            except Exception:
                pass
