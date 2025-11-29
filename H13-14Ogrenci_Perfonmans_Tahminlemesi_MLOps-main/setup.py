from typing import List

from setuptools import find_packages, setup

HYPEN_E_DOT = '-e .'
def get_requirements(file_path:str)->List[str]: #Parametre string tipinde olucak ve bu fonskiyon içi string tipinde elemanları olan bir liste döndürecektir
    """
    Bu fonksiyon gereken kütüphaneleri listenin elemanı olarak dönecektir.
    """
    with open(file_path) as file_obj:
        requirements = file_obj.readlines()
        requirements = [req.replace('\n',"") for req in requirements]
        if HYPEN_E_DOT in requirements:
            requirements.remove(HYPEN_E_DOT)

    return requirements
setup(
    name='ogrencitahminleme',
    version = '0.0.1',
    author='emre',
    author_email='yunusemrek558@gmail.com',
    packages=find_packages(),
    install_requires = get_requirements('/workspaces/-renci_Perfonmans_Tahminlemesi_MLOps_Deneme2/requirements.txt')
)