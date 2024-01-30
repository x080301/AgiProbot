import wandb
import subprocess
import socket


def find_one_block(lines):
    run_name = None
    for i, line in enumerate(lines):
        if 'wandb.name=' in line:
            run_name = line.split('wandb.name=')[1].split("'")[1]
            if 'test_' in line:
                run_name = run_name + '_test'
        if '--------' in line:
            break

    lines_in_one_block = lines[:i]
    if i < len(lines) - 1:
        remaining_lines = lines[(i + 1):]
    else:
        remaining_lines = None

    return run_name, lines_in_one_block, remaining_lines


def get_trained_runs():
    wandb.login(key='d548a03793d0947aea18838d6b45447d207ec072')

    project_name = "APESv3"
    entity_name = "fuhaodsst"

    api = wandb.Api()
    runs = api.runs(f"{entity_name}/{project_name}")

    trained_runs = []
    for run in runs:
        trained_runs.append(run.name[17:])

    return trained_runs


def find_one_block_to_train(lines, trained_runs):
    trained_runs.extend(get_trained_runs())
    remaining_lines = lines
    one_block_to_train = None
    while remaining_lines is not None:
        run_name, lines_in_one_block, remaining_lines = find_one_block(remaining_lines)
        if run_name not in trained_runs:
            one_block_to_train = lines_in_one_block
            break

    if one_block_to_train is None:
        run_name = None

    return run_name, one_block_to_train


def run_cmds(run_name, cmd_block, running_gpu):
    for single_line in cmd_block:
        if 'which_gpu=[]' in single_line:
            single_line.replace('which_gpu=[]', running_gpu)
        single_line = single_line.strip('\n')
        result = subprocess.run(single_line, shell=True, text=True, stdout=None,  # subprocess.PIPE,
                                stderr=subprocess.PIPE)
        if result.returncode != 0:
            with open(f'train_logs/{run_name}.txt', 'w') as file:
                file.writelines(result.stderr)
            return False
    return True


if __name__ == '__main__':
    running_gpu = {'server': 'which_gpu=[2]',  # which_gpu=[]
                   'work_station': 'which_gpu=[0,1]'}

    hostname = socket.gethostname()
    if 'iesservergpu' in hostname:
        running_device = 'server'
    else:
        running_device = 'work_station'

    trained_runs = []
    while True:

        with open('training_in_line.txt', 'r') as file:
            lines = file.readlines()

        run_name, cmd_block = find_one_block_to_train(lines, trained_runs)
        if run_name is None:
            print('training finished')
            break
        else:
            trained_runs.append(run_name)
            if running_device == 'work_station' or '_test' not in run_name:
                print(run_cmds(run_name, cmd_block, running_gpu[running_device]))
