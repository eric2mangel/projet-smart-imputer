import logging

import pandas as pd
import pandera.pandas as pa

logger = logging.getLogger(__name__)

# Schéma du "contrat de données" :
# Ici, seules les colonnes critiques son décrites
# Pas besoin de lister toutes les colonnes du fichier :
# strict=False laisse passer les autres sans les valider.
raw_data_schema = pa.DataFrameSchema(
    columns={
        "age": pa.Column(float, nullable=True, checks=pa.Check.in_range(0, 120)),
        "revenu_annuel": pa.Column(float, nullable=True, checks=pa.Check.ge(0)),
        "default": pa.Column(int, checks=pa.Check.isin([0, 1])),
    },
    strict=False,
)


def validate_data(df: pd.DataFrame) -> pd.DataFrame:
    """Valide le DataFrame par rapport au schéma métier ci-dessus.

    Contrairement aux valeurs manquantes (attendues et gérées plus loin par
    l'imputation), une valeur aberrante comme un âge de -5 ou 300 ans est le
    signe d'une erreur de saisie ou d'extraction. Il est préférable arrêter le
    pipeline ici, avec un message clair, plutôt que de laisser cette valeur
    fausser silencieusement l'imputation ou le modèle.
    """
    try:
        df_valide = raw_data_schema.validate(df, lazy=True)
        logger.info(
            "Validation pandera OK : %s lignes conformes au schéma", len(df_valide)
        )
        return df_valide
    except pa.errors.SchemaErrors as e:
        logger.error("Validation pandera échouée :\n%s", e.failure_cases)
        raise
