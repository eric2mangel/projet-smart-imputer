# Pour éviter des imports à rallonge dans le script principal,
# on rend les fonctions directement accessibles depuis le package smart_imputer.
from .cleaner import (
    clean_column_names,
    load_data,
    remove_duplicates,
    remove_empty_columns,
)
from .imputer import impute_missing_values
from .pipeline import prepare_data
from .schema import validate_data
