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

    "python train_shapenet.py train.epochs=200 train.ddp.which_gpu=[] datasets=shapenet_AnTao350M usr_config=configs/seg_boltzmannT01_bin2.yaml wandb.name='Shapenet_Token_Std_boltzmann_T01_bin2_1'",
    "python train_shapenet.py train.epochs=200 train.ddp.which_gpu=[] datasets=shapenet_AnTao350M usr_config=configs/seg_boltzmannT01_bin2.yaml wandb.name='Shapenet_Token_Std_boltzmann_T01_bin2_2'",
    "python train_shapenet.py train.epochs=200 train.ddp.which_gpu=[] datasets=shapenet_AnTao350M usr_config=configs/seg_boltzmannT01_bin2.yaml wandb.name='Shapenet_Token_Std_boltzmann_T01_bin2_3'",
    "python train_shapenet.py train.epochs=200 train.ddp.which_gpu=[] datasets=shapenet_AnTao350M usr_config=configs/seg_boltzmannT01_bin4.yaml wandb.name='Shapenet_Token_Std_boltzmann_T01_bin4_1'",
    "python train_shapenet.py train.epochs=200 train.ddp.which_gpu=[] datasets=shapenet_AnTao350M usr_config=configs/seg_boltzmannT01_bin4.yaml wandb.name='Shapenet_Token_Std_boltzmann_T01_bin4_2'",
    "python train_shapenet.py train.epochs=200 train.ddp.which_gpu=[] datasets=shapenet_AnTao350M usr_config=configs/seg_boltzmannT01_bin4.yaml wandb.name='Shapenet_Token_Std_boltzmann_T01_bin4_3'",
    "python train_shapenet.py train.epochs=200 train.ddp.which_gpu=[] datasets=shapenet_AnTao350M usr_config=configs/seg_boltzmannT01_bin6.yaml wandb.name='Shapenet_Token_Std_boltzmann_T01_bin6_1'",
    "python train_shapenet.py train.epochs=200 train.ddp.which_gpu=[] datasets=shapenet_AnTao350M usr_config=configs/seg_boltzmannT01_bin6.yaml wandb.name='Shapenet_Token_Std_boltzmann_T01_bin6_2'",
    "python train_shapenet.py train.epochs=200 train.ddp.which_gpu=[] datasets=shapenet_AnTao350M usr_config=configs/seg_boltzmannT01_bin6.yaml wandb.name='Shapenet_Token_Std_boltzmann_T01_bin6_3'",
    "python train_shapenet.py train.epochs=200 train.ddp.which_gpu=[] datasets=shapenet_AnTao350M usr_config=configs/seg_boltzmannT01_bin8.yaml wandb.name='Shapenet_Token_Std_boltzmann_T01_bin8_1'",
    "python train_shapenet.py train.epochs=200 train.ddp.which_gpu=[] datasets=shapenet_AnTao350M usr_config=configs/seg_boltzmannT01_bin8.yaml wandb.name='Shapenet_Token_Std_boltzmann_T01_bin8_2'",
    "python train_shapenet.py train.epochs=200 train.ddp.which_gpu=[] datasets=shapenet_AnTao350M usr_config=configs/seg_boltzmannT01_bin8.yaml wandb.name='Shapenet_Token_Std_boltzmann_T01_bin8_3'",
    "python train_shapenet.py train.epochs=200 train.ddp.which_gpu=[] datasets=shapenet_AnTao350M usr_config=configs/seg_boltzmannT01_bin10.yaml wandb.name='Shapenet_Token_Std_boltzmann_T01_bin10_1'",
    "python train_shapenet.py train.epochs=200 train.ddp.which_gpu=[] datasets=shapenet_AnTao350M usr_config=configs/seg_boltzmannT01_bin10.yaml wandb.name='Shapenet_Token_Std_boltzmann_T01_bin10_2'",
    "python train_shapenet.py train.epochs=200 train.ddp.which_gpu=[] datasets=shapenet_AnTao350M usr_config=configs/seg_boltzmannT01_bin10.yaml wandb.name='Shapenet_Token_Std_boltzmann_T01_bin10_3'"

]
trained_list = get_trained_runs()

subprocess.run('nvidia-smi', shell=True, text=True, stdout=None, stderr=subprocess.PIPE)

for testline in test_list:
    testline_1 = testline.split("wandb.name='")[-1].split("'")[0]
    config = testline.split("usr_config=")[-1].split(" wandb.name=")[0]
    # print(config)
    # print(testline_1)
    for trained in trained_list:
        if testline_1 in trained and trained[-1] == testline_1[-1]:  # len(trained.split(testline_1))==1:

            if 'Modelnet' in trained:
                test_cmd = f"python test_modelnet.py datasets=modelnet_AnTao420M usr_config={config} wandb.name='{trained}' test.ddp.which_gpu=[0,1]"
            elif 'Shapenet' in trained:
                test_cmd = f"python test_shapenet.py datasets=shapenet_AnTao350M usr_config={config} wandb.name='{trained}' test.ddp.which_gpu=[0,1]"
            else:
                raise NotImplementedError
            # subprocess.run(test_cmd, shell=True, text=True, stdout=None,  # subprocess.PIPE,
            #                stderr=subprocess.PIPE)

    print(test_cmd)
    # if :
    #     print(trained)
    # print(testline)
