import logging
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
import yaml  # Pour charger le fichier de configuration

# Ajoute le dossier 'src' au chemin de recherche de Python
sys.path.append(str(Path(__file__).parent / "src"))


from smart_imputer import (
    prepare_data,
    impute_missing_values,
)

# Import direct de la fonction propre
from smart_imputer.config import load_config
from smart_imputer.logging_config import setup_logging

########### Chargement des informations secrètes
load_dotenv() # Lecture du fichier

# Récupération des clés secrètes
api_key = os.getenv("API_KEY")
db_password = os.getenv("DATABASE_PASSWORD")

# Logger dédié à ce script (le nom "main" apparaîtra dans les logs)
logger = logging.getLogger("main")


if __name__ == "__main__":
    # 0. Configuration du logging (console + fichier logs/smart_imputer.log)
    setup_logging()

    # 1. Chargement de la configuration externe
    chemin_yaml = Path("config") / "config.yaml"
    config = load_config(chemin_yaml)
    logger.info("Configuration chargée depuis %s", chemin_yaml)

    # 2. Récupération des constantes et chemins depuis le dictionnaire de config
    FILE_PATH = config.paths.raw_data
    OUTPUT_PATH = config.paths.processed_data
    MODELS_PATH = config.paths.models    # <-- AJOUT
    DELETION_THRESHOLD = config.parameters.deletion_threshold
    R2_THRESHOLD = config.parameters.r2_threshold
    RF_N_ESTIMATORS = config.model.n_estimators
    RF_RANDOM_STATE = config.model.random_state

    # 3. Pipeline
    # Chargement + nettoyage + validation, mis en cache par joblib.Memory
    # (voir smart_imputer/pipeline.py).
    df = prepare_data(FILE_PATH)
    logger.info("Données prêtes (%s) - shape : %s", FILE_PATH, df.shape)

    MODELS_PATH.mkdir(parents=True, exist_ok=True)   # <-- AJOUTÉ
    df = impute_missing_values(
        df, 
        deletion_threshold=DELETION_THRESHOLD, 
        r2_threshold=R2_THRESHOLD,
        nb_estimators=RF_N_ESTIMATORS,
        random_value=RF_RANDOM_STATE,
        models_dir=MODELS_PATH,              # <-- AJOUTÉ
    )

    logger.info("Pipeline terminé - shape finale : %s", df.shape)

    # Sauvegarde dans le dossier processed
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)
    logger.info("Données sauvegardées dans %s", OUTPUT_PATH)