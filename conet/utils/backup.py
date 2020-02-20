# from brats.config import exp_config
import shutil
import os

#https://stackoverflow.com/questions/1855095/how-to-create-a-zip-archive-of-a-directory
def backup_project_as_zip(project_dir, zip_file):
    assert(os.path.isdir(project_dir))
    assert(os.path.isdir(os.path.dirname(zip_file)))
    print(f'Project path => {project_dir}, backup to {zip_file}')
    shutil.make_archive(zip_file.replace('.zip',''), 'zip', project_dir)
    pass