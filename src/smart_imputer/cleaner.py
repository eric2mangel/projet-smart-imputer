import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


def load_data(chemin: Path) -> pd.DataFrame:
    """
    Charge le fichier CSV depuis le chemin donné.
    La gestion des erreurs ajoutée.
    """
    try:
        df = pd.read_csv(chemin)
        logger.info("Lecture du fichier CSV : %s", chemin)
        return df
    except FileNotFoundError:
        logger.error(f"Fichier introuvable : {chemin}")
        raise
    except pd.errors.EmptyDataError:
        logger.error(f"Fichier vide : {chemin}")
        raise
    except Exception as e:
        logger.error(f"Erreur inattendue : {e}")
        raise


def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Normalise les noms de colonnes en snake_case minuscule."""
    df.columns = [
        c.strip().lower().replace(" ", "_").replace("-", "_") for c in df.columns
    ]
    return df


def remove_empty_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Supprime les colonnes entièrement vides."""
    df_clean = df.dropna(axis=1, how="all")
    nb_supprimees = df.shape[1] - df_clean.shape[1]
    if nb_supprimees > 0:
        logger.info("%s colonne(s) entièrement vide(s) supprimée(s)", nb_supprimees)
    return df_clean


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Supprime les lignes en double."""
    df_clean = df.drop_duplicates()
    nb_doublons = df.shape[0] - df_clean.shape[0]
    if nb_doublons > 0:
        logger.info("%s ligne(s) dupliquée(s) supprimée(s)", nb_doublons)
    return df_clean
