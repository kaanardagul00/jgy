import os
from box.exceptions import BoxValueError
import yaml  # type: ignore
import json
import joblib # type: ignore
from ensure import ensure_annotations
from box import ConfigBox
from pathlib import Path
from typing import Any
import base64
import pickle
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import r2_score


@ensure_annotations
def read_yaml(path_to_yaml: Path) -> ConfigBox:
    """reads yaml file and returns

    Args:
        path_to_yaml (str): path like input

    Raises:
        ValueError: if yaml file is empty
        e: empty file

    Returns:
        ConfigBox: ConfigBox type
    """
    try:
        with open(path_to_yaml) as yaml_file:
            content = yaml.safe_load(yaml_file)
            return ConfigBox(content)
    except BoxValueError:
        raise ValueError("yaml file is empty")
    except Exception as e:
        raise e
    


@ensure_annotations
def create_directories(path_to_directories: list, verbose=True):
    """create list of directories

    Args:
        path_to_directories (list): list of path of directories
        ignore_log (bool, optional): ignore if multiple dirs is to be created. Defaults to False.
    """
    for path in path_to_directories:
        os.makedirs(path, exist_ok=True)


@ensure_annotations
def get_size(path: Path) -> str:
    """get size in KB

    Args:
        path (Path): path of the file

    Returns:
        str: size in KB
    """
    size_in_kb = round(os.path.getsize(path)/1024)
    return f"~ {size_in_kb} KB"
@ensure_annotations
def save_object(file_path:Path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise e

def evaluate_model(X_train,y_train,X_test,y_test,models,param):
    try:
        report = {}
        for i in range(0,len(list(models))): # 0-7 döngü başlatır

            model = list(models.values())[i]  # i == 0 iken RandomForestRegressor() çalışır.
            para = param[list(models.keys())[i]]
            rc = RandomizedSearchCV(model,para,cv=3)

            rc.fit(X_train, y_train) # Bu search algoritmasını Train datalar üzerinden çalıştırdım
            model.set_params(**rc.best_params_) # Yukarıda çalışan search algoritmasının bulduğu en iyi parametreleri aldım
            model.fit(X_train, y_train) # En iyi parametrelerle eğitim yaptık

            y_test_pred = model.predict(X_test) # Tahmin değerlerini aldık

            test_model_score = r2_score(y_test,y_test_pred)


            report[list(models.keys())[i]] = test_model_score

        return report

    except Exception as e:
        raise e