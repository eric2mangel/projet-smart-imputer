from pathlib import Path

import pandas as pd
from joblib import Memory

from .cleaner import (
    clean_column_names,
    load_data,
    remove_duplicates,
    remove_empty_columns,
)
from .schema import validate_data

# Cache disque pour les fonctions coûteuses : le résultat n'est recalculé
# que si les arguments (ici, le chemin du fichier) ou le code de la fonction
# changent. Le dossier cache/ est créé automatiquement au premier appel.
memory = Memory(location="cache/", verbose=0)


@memory.cache
def prepare_data(chemin: Path) -> pd.DataFrame:
    """Charge, nettoie et valide les données brutes.

    Regroupe tout le pré-traitement (avant l'imputation)
    Tant que le fichier source ne change pas,
    un deuxième appel restitue le résultat instantanément,
    sans relireni renettoyer ni revalider le CSV.
    """
    df = load_data(chemin)
    df = clean_column_names(df)
    df = remove_empty_columns(df)
    df = remove_duplicates(df)
    return validate_data(df)
