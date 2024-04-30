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
        single_line = single_line.strip('\n')
        if 'which_gpu=[]' in single_line:
            single_line = single_line.replace('which_gpu=[]', running_gpu)

        result = subprocess.run(single_line, shell=True, text=True, stdout=None,  # subprocess.PIPE,
                                stderr=subprocess.PIPE)
        if result.returncode != 0:
            with open(f'train_logs/{run_name}.txt', 'w') as file:
                file.writelines(result.stderr)
            return result.stderr
    return True


if __name__ == '__main__':
    num_gpus = 2

    result = subprocess.run(['nvidia-smi', '--query-gpu=index,power.draw,memory.used,utilization.gpu,memory.free',
                             '--format=csv,noheader,nounits'], stdout=subprocess.PIPE, text=True)
    output = result.stdout.strip().split('\n')
    gpu_info = []
    for line in output:
        gpu_index, power_draw, memory_used, gpu_util, memory_free = line.split(',')
        gpu_info.append({'GPU Index': int(gpu_index), 'Power Draw (W)': float(power_draw),
                         'Memory Used (MB)': float(memory_used), 'GPU Utilization (%)': float(gpu_util),
                         'Memory Free (MB)': float(memory_free)})

    available_gpus = []
    for gpu in gpu_info:
        print(f'GPU {gpu["GPU Index"]} Utilization (%): {gpu["GPU Utilization (%)"]}, Memory Used (MB): {gpu["Memory Free (MB)"]}')
        if gpu['GPU Utilization (%)'] > 20 or gpu['Memory Free (MB)'] < 11000:
            continue
        else:
            available_gpus.append(gpu['GPU Index'])
    if len(available_gpus) < num_gpus:
        raise ValueError(
            f"Not enough available GPUs! Available GPUs: {available_gpus}, Required number of GPUs: {num_gpus}")
    else:
        if len(available_gpus) == 2:
            running_gpu = f'which_gpu=[{available_gpus[0]}]'
        else:
            running_gpu = available_gpus[:num_gpus]
            running_gpu = f'which_gpu=[{available_gpus[0]},{available_gpus[1]}]'

    # running_gpu = {'server_d': 'which_gpu=[2,3]',  # which_gpu=[]
    #                'work_station': 'which_gpu=[0,1]',
    #                'server_a': 'which_gpu=[0,1]'}
    hostname = socket.gethostname()
    if 'iesservergpu-d' in hostname:
        running_device = 'server_d'
    elif 'iesservergpu-a' in hostname:
        running_device = 'server_a'
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
                print(run_cmds(run_name, cmd_block, running_gpu))
