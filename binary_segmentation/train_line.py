import os

if __name__ == "__main__":

    file_list = []

    while True:
        executable_str = None
        for txt_file_name in sorted(os.listdir('train_line')):
            if txt_file_name not in file_list:
                file_list.append(txt_file_name)
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
