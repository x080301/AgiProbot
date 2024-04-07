import wandb
import subprocess


def get_trained_runs():
    wandb.login(key='d548a03793d0947aea18838d6b45447d207ec072')

    project_name = "APESv3"
    entity_name = "fuhaodsst"

    api = wandb.Api()
    runs = api.runs(f"{entity_name}/{project_name}")

    trained_runs = []
    for run in runs:
        trained_runs.append(run.name)  # [17:])

    return trained_runs


test_list = [
    "python train_modelnet.py train.epochs=200 train.ddp.which_gpu=[] datasets=modelnet_AnTao420M usr_config=configs/boltzmannT01.yaml wandb.name='Modelnet_Token_Std_boltzmann_T01_norm_5'",
    "python train_modelnet.py train.epochs=200 train.ddp.which_gpu=[] datasets=modelnet_AnTao420M usr_config=configs/boltzmannT01.yaml wandb.name='Modelnet_Token_Std_boltzmann_T01_norm_6'",
    "python train_modelnet.py train.epochs=200 train.ddp.which_gpu=[] datasets=modelnet_AnTao420M usr_config=configs/boltzmannT01.yaml wandb.name='Modelnet_Token_Std_boltzmann_T01_norm_7'",
    "python train_modelnet.py train.epochs=200 train.ddp.which_gpu=[] datasets=modelnet_AnTao420M usr_config=configs/boltzmannT01.yaml wandb.name='Modelnet_Token_Std_boltzmann_T01_norm_8'",
    "python train_modelnet.py train.epochs=200 train.ddp.which_gpu=[] datasets=modelnet_AnTao420M usr_config=configs/boltzmannT01.yaml wandb.name='Modelnet_Token_Std_boltzmann_T01_norm_9'",
    "python train_modelnet.py train.epochs=200 train.ddp.which_gpu=[] datasets=modelnet_AnTao420M usr_config=configs/boltzmannT01.yaml wandb.name='Modelnet_Token_Std_boltzmann_T01_norm_10'"]

trained_list = get_trained_runs()

subprocess.run('nvidia-smi', shell=True, text=True, stdout=None, stderr=subprocess.PIPE)

for testline in test_list:
    testline_1 = testline.split("wandb.name='")[-1].split("'")[0]
    config = testline.split("usr_config=")[-1].split(" wandb.name=")[0]
    # print(config)
    # print(testline_1)
    for trained in trained_list:
        if testline_1 in trained and trained[-1] == testline_1[-1]:  # len(trained.split(testline_1))==1:
            test_cmd = f"python test_modelnet.py datasets=modelnet_AnTao420M usr_config={config} wandb.name='{trained}' test.ddp.which_gpu=[0,1]"

            # subprocess.run(test_cmd, shell=True, text=True, stdout=None,  # subprocess.PIPE,
            #                stderr=subprocess.PIPE)

    print(test_cmd)
    # if :
    #     print(trained)
    # print(testline)
