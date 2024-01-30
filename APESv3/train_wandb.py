import subprocess
import socket
import wandb


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



if __name__ == '__main__':

    trained_runs = []
    do_next_training = True

    while do_next_training:
        # read trainings
        do_next_training = False
        with open('training_in_line.txt', 'r') as file:
            lines = file.readlines()

        # find next training
        cmds_to_run = []
        runing_name = None
        for single_line in lines:
            single_line = single_line.strip('\n')
            if '-------' in single_line:
                if runing_name is not None:
                    break

                cmds_to_run = []
                runing_name = None

        do_next_training = True
        # do next training
