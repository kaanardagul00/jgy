import os 
from pathlib import Path
import logging

logging.basicConfig(level = logging.INFO, format='[%(asctime)s]: %(message)s')

project_name = 'ÖğrenciTahminleme'

list_of_files = [
    f"src/{project_name}/__init__.py",
    f"src/{project_name}/components/__init__.py",
    f"src/{project_name}/utils/__init__.py",
    f"src/{project_name}/utils/common.py",
    f"src/{project_name}/config/__init__.py",
    f"src/{project_name}/config/configration.py",
    f"src/{project_name}/pipeline/__init__.py",
    f"src/{project_name}/entity/__init__.py",
    f"src/{project_name}/entity/config_entity.py",
    f"src/{project_name}/constants/__init__.py",
    "config/config.yaml",
    "dvc.yaml",
    "params.yaml",
    "requirements.txt",
    "setup.py",
    "research/deneme.ipynb",
    "templates/index.html",
    "main.py"
]

for filepath in list_of_files:
    filepath = Path(filepath) # yukarıdaki string tipleri bir Path tipine yani dosya yolu tipine çeviriyor.
    filedir,filename = os.path.split(filepath)

    if filedir != "": # eğer boş ise False (Boş Değilse True yani içinde dosya varsa True)
        os.makedirs(filedir,exist_ok=True) # exist_ok= True Eğer bu isimle bir yol oluşuturulduysa ve tekrardan os.makedirs() bu çalıştırılsa hata verilmesini sağlar
        logging.info(f'Oluşturalan dosyanın yolu: {filedir} ve oluşturulan dosyanın ismi: {filename}')

    if (not os.path.exists(filepath)) or (os.path.getsize(filepath) == 0):
        with open(filepath, 'w') as f:
            pass
            logging.info(f'Oluşturulan boş dosyanın yolu: {filepath}')

    else:
        logging.info(f"{filename} isimli dosya zaten mevcut tekrardan oluşturulamadı.")