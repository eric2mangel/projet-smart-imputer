import logging
from pathlib import Path  # Ajout

import joblib  # Ajout
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

# Un logger par module : son nom (ex. "smart_imputer.imputer") permet de savoir
# précisément d'où vient chaque message dans les logs.
logger = logging.getLogger(__name__)


def impute_missing_values(
    df: pd.DataFrame,
    deletion_threshold: float = 40.0,
    r2_threshold: float = 0.6,
    nb_estimators: int = 50,
    random_value: int = 42,
    models_dir: Path | None = None,  # Ajout
) -> pd.DataFrame:
    """Impute les valeurs manquantes (médiane puis Random Forest si R2 suffisant).

    Args:
        df: DataFrame à nettoyer.
        deletion_threshold: Taux (%) de manquants déclenchant la suppression.
        r2_threshold: R2 minimum pour activer l'imputation intelligente.
        nb_estimators : Nombre d'estimateurs du RandomForest
        random_value : Valeur du random state
        models_dir: Dossier où sauvegarder/charger les modèles entraînés
            (joblib). Si None, aucune persistance n'est effectuée : on
            entraîne un modèle à chaque appel, comme avant.

    Returns:
        DataFrame avec valeurs manquantes imputées ou colonnes supprimées.
    """
    cols_num = df.select_dtypes(include="number").columns
    for c in cols_num:
        missing_count = df[c].isnull().sum()
        if missing_count == 0:
            continue

        missing_pct = (missing_count / len(df)) * 100
        logger.info("%s -> %s%% manquants", c, round(missing_pct, 1))

        if missing_pct > deletion_threshold:
            logger.warning("%s : trop de manquants -> suppression de la colonne", c)
            df = df.drop(columns=[c])
            continue

        # Imputation par la médiane
        val_median = df[c].median()
        df[c] = df[c].fillna(val_median)

        # Tentative d'imputation intelligente
        if missing_pct < 20:
            feats = [x for x in cols_num if x != c and x in df.columns]
            train_df = df[df[c].notnull()]

            if len(train_df) > 15 and len(feats) > 0:
                # Chemin du modèle persisté pour cette colonne, si un dossier
                # de modèles a été fourni (sinon on ré-entraîne à chaque fois).
                model_path = models_dir / f"rf_{c}.joblib" if models_dir else None

                if model_path and model_path.exists():
                    rf = joblib.load(model_path)
                    logger.info(
                        "%s : modèle rechargé depuis %s (pas de ré-entraînement)",
                        c,
                        model_path,
                    )
                else:
                    rf = RandomForestRegressor(
                        n_estimators=nb_estimators, random_state=random_value
                    )
                    rf.fit(train_df[feats], train_df[c])
                    if model_path:
                        model_path.parent.mkdir(parents=True, exist_ok=True)
                        joblib.dump(rf, model_path)
                        logger.info(
                            "%s : modèle entraîné puis sauvegardé dans %s",
                            c,
                            model_path,
                        )

                pred_train = rf.predict(train_df[feats])
                r2 = r2_score(train_df[c], pred_train)
                logger.debug("%s : R2 = %s", c, round(r2, 3))

                if r2 > r2_threshold:
                    missing_rows = df[df[c].isnull()]
                    if len(missing_rows) > 0:
                        preds = rf.predict(missing_rows[feats])
                        df.loc[df[c].isnull(), c] = preds
                        logger.info(
                            "%s : imputation intelligente (RandomForest) appliquée", c
                        )
                else:
                    logger.info(
                        "%s : R2 trop faible (%s <= %s) -> on garde uniquement l'imputation par la médiane",
                        c,
                        round(r2, 3),
                        r2_threshold,
                    )

    return df
