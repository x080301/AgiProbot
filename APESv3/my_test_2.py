import wandb


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
    "python train_modelnet.py train.epochs=200 train.ddp.which_gpu=[0,1] datasets=modelnet_AnTao420M usr_config=configs/token_nonaveragebins_std_cls_topM.yaml wandb.name='Modelnet_Token_Std_topM'",
    "python train_modelnet.py train.epochs=200 train.ddp.which_gpu=[0,1] datasets=modelnet_AnTao420M usr_config=configs/token_nonaveragebins_std_cls_topM.yaml wandb.name='Modelnet_Token_Std_topM_1'",
    "python train_modelnet.py train.epochs=200 train.ddp.which_gpu=[0,1] datasets=modelnet_AnTao420M usr_config=configs/token_nonaveragebins_std_cls_topM.yaml wandb.name='Modelnet_Token_Std_topM_2'",
    "python train_modelnet.py train.epochs=200 train.ddp.which_gpu=[0,1] datasets=modelnet_AnTao420M usr_config=configs/token_nonaveragebins_std_cls_topM.yaml wandb.name='Modelnet_Token_Std_topM_3'",
    "python train_modelnet.py train.epochs=200 train.ddp.which_gpu=[0,1] datasets=modelnet_AnTao420M usr_config=configs/token_nonaveragebins_std_cls_uniform.yaml wandb.name='Modelnet_Token_Std_uniform'",
    "python train_modelnet.py train.epochs=200 train.ddp.which_gpu=[0,1] datasets=modelnet_AnTao420M usr_config=configs/token_nonaveragebins_std_cls_uniform.yaml wandb.name='Modelnet_Token_Std_uniform_1'",
    "python train_modelnet.py train.epochs=200 train.ddp.which_gpu=[0,1] datasets=modelnet_AnTao420M usr_config=configs/token_nonaveragebins_std_cls_uniform.yaml wandb.name='Modelnet_Token_Std_uniform_2'",
    "python train_modelnet.py train.epochs=200 train.ddp.which_gpu=[0,1] datasets=modelnet_AnTao420M usr_config=configs/token_nonaveragebins_std_cls_uniform.yaml wandb.name='Modelnet_Token_Std_uniform_3'",
    "python train_modelnet.py train.epochs=200 train.ddp.which_gpu=[0,1] datasets=modelnet_AnTao420M usr_config=configs/token_nonaveragebins_std_cls_boltmannT05.yaml wandb.name='Modelnet_Token_Std_boltmannT05_1'",
    "python train_modelnet.py train.epochs=200 train.ddp.which_gpu=[0,1] datasets=modelnet_AnTao420M usr_config=configs/token_nonaveragebins_std_cls_boltmannT05.yaml wandb.name='Modelnet_Token_Std_boltmannT05_2'",
    "python train_modelnet.py train.epochs=200 train.ddp.which_gpu=[0,1] datasets=modelnet_AnTao420M usr_config=configs/token_nonaveragebins_std_cls_boltmannT05.yaml wandb.name='Modelnet_Token_Std_boltmannT05_3'",
    "python train_modelnet.py train.epochs=200 train.ddp.which_gpu=[0,1] datasets=modelnet_AnTao420M usr_config=configs/token_nonaveragebins_std_cls_boltmannT05.yaml wandb.name='Modelnet_Token_Std_boltmannT05_4'",
    "python train_modelnet.py train.epochs=200 train.ddp.which_gpu=[0,1] datasets=modelnet_AnTao420M usr_config=configs/token_nonaveragebins_std_cls_boltmannT02.yaml wandb.name='Modelnet_Token_Std_boltmannT02_1'",
    "python train_modelnet.py train.epochs=200 train.ddp.which_gpu=[0,1] datasets=modelnet_AnTao420M usr_config=configs/token_nonaveragebins_std_cls_boltmannT02.yaml wandb.name='Modelnet_Token_Std_boltmannT02_2'",
    "python train_modelnet.py train.epochs=200 train.ddp.which_gpu=[0,1] datasets=modelnet_AnTao420M usr_config=configs/token_nonaveragebins_std_cls_boltmannT02.yaml wandb.name='Modelnet_Token_Std_boltmannT02_3'",
    "python train_modelnet.py train.epochs=200 train.ddp.which_gpu=[0,1] datasets=modelnet_AnTao420M usr_config=configs/token_nonaveragebins_std_cls_boltmannT02.yaml wandb.name='Modelnet_Token_Std_boltmannT02_4'"]

trained_list = get_trained_runs()

for testline in test_list:
    testline_1 = testline.split("wandb.name='")[-1].split("'")[0]
    config = testline.split("usr_config=")[-1].split(" wandb.name=")[0]
    # print(config)
    # print(testline_1)
    for trained in trained_list:
        if testline_1 in trained and trained[-1]==testline_1[-1]:#len(trained.split(testline_1))==1:
            test_cmd=f"test_modelnet.py datasets=modelnet_AnTao420M usr_config={config} wandb.name='{trained}' test.ddp.which_gpu=[0,1]"

    print(test_cmd)
        # if :
        #     print(trained)
    # print(testline)