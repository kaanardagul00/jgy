import os

import sys
from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor
)

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score # Yüzdelik olarak modelin ne kadar doğru çalıştığını gösterir
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from src.ÖğrenciTahminleme import logger
from dataclasses import dataclass
from src.ÖğrenciTahminleme.utils.common import get_size,save_object,evaluate_model

import yaml
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import r2_score
from src.ÖğrenciTahminleme.utils.common import read_yaml  # Mevcut read_yaml fonksiyonunu kullanıyoruz
from src.ÖğrenciTahminleme.entity.config_entity import ModelTrainerConfig
from src.ÖğrenciTahminleme.constants import *

class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig,params_filepath = PARAMS_FILE_PATH):
        self.config = config
        self.params = read_yaml(PARAMS_FILE_PATH)  # params path config'den okunur

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logger.info("Train ve test verisiyle model eğitimi başlatılıyor.")
            
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1],
            )

            models = {
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "CatBoosting Regressor": CatBoostRegressor(verbose=False),
                "AdaBoost Regressor": AdaBoostRegressor(),
            }

            # Parametreleri params.yaml dosyasından oku
            params = self.params

            model_report = {}
            for model_name, model in models.items():
                if model_name in params:
                    param_grid = params[model_name]
                    if param_grid:  # Eğer model için parametre tanımlandıysa
                        logger.info(f"{model_name} için RandomizedSearchCV başlatılıyor.")
                        search = RandomizedSearchCV(
                            estimator=model,
                            param_distributions=param_grid,
                            scoring="r2",
                            n_iter=10,  # Rastgele 10 kombinasyonu değerlendir
                            cv=3,  # 3-fold cross-validation
                            random_state=42,
                            n_jobs=-1
                        )
                        search.fit(X_train, y_train)
                        best_model = search.best_estimator_
                        model_report[model_name] = best_model.score(X_test, y_test)
                    else:
                        logger.info(f"{model_name} için parametre yok, varsayılan model kullanılıyor.")
                        model.fit(X_train, y_train)
                        model_report[model_name] = model.score(X_test, y_test)
                else:
                    logger.warning(f"{model_name} için params.yaml'de parametre bulunamadı.")

            # En iyi modeli bul
            best_model_name = max(model_report, key=model_report.get)
            best_model_score = model_report[best_model_name]
            best_model = models[best_model_name]

            if best_model_score < 0.6:
                raise Exception("Hiçbir model yeterli performansı sağlayamadı.")

            logger.info(f"En iyi model: {best_model_name} (R2: {best_model_score})")

            # Modeli kaydet
            save_object(
                file_path=self.config.model_file / "best_model.pkl",
                obj=best_model
            )

            predicted = best_model.predict(X_test)
            r2_square = r2_score(y_test, predicted)
            return best_model, r2_square

        except Exception as e:
            logger.error(f"Model eğitimi sırasında hata oluştu: {e}")
            raise