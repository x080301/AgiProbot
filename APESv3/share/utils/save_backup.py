import os
import subprocess
import zipfile
from datetime import datetime

def zip_folder(output_path, folder_path):
    if os.path.exists(output_path):
        os.remove(output_path)
    with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, _, files in os.walk(folder_path):
            for file in files:
                file_path = os.path.join(root, file)
                rel_path = os.path.relpath(file_path, folder_path)
                zipf.write(file_path, rel_path)

# storage and backup
def save_backup(artifacts_path, zip_file_path, backup_path):
    zip_folder(zip_file_path, artifacts_path)
    if os.path.exists(backup_path):
        current_date = subprocess.check_output(['date'], universal_newlines=True, shell=True)
        current_date = datetime.strptime(current_date.strip(), '%a %b %d %H:%M:%S %Z %Y')
        current_date = current_date.strftime('%Y-%m-%d_%H-%M-%S')
        os.system(f'mv {artifacts_path} {artifacts_path}_{current_date}')
        zip_file_path_new = os.path.splitext(zip_file_path)[0] + "_" + current_date + ".zip"
        os.system(f'mv {zip_file_path} {zip_file_path_new}')
        artifacts_path = f'{artifacts_path}_{current_date}'
        zip_file_path = zip_file_path_new
    os.system(f'cp -r {artifacts_path} /data/users/wan/artifacts')
    os.system(f'cp {zip_file_path} /data/users/wan/backup_zip')
    os.system(f'rm {zip_file_path}')