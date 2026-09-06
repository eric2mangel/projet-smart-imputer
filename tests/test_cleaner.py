import numpy as np
import pandas as pd
import pytest

from smart_imputer.cleaner import clean_column_names, remove_empty_columns
from smart_imputer.imputer import impute_missing_values


@pytest.fixture
def df_sale():
    """DataFrame avec noms mal formés et valeurs manquantes."""
    return pd.DataFrame({
        "Age": [25, 30, np.nan, 40],
        "Revenu Annuel": [50000, np.nan, 60000, 70000],
        "code-postal": ["75001", "69001", "13001", "33000"],
        "colonne_vide": [None, None, None, None],
    })


def test_nettoyage_noms_colonnes(df_sale):
    df = clean_column_names(df_sale)
    assert list(df.columns) == ["age", "revenu_annuel", "code_postal", "colonne_vide"]


def test_suppression_colonne_vide(df_sale):
    df = remove_empty_columns(df_sale)
    assert "colonne_vide" not in df.columns
    assert "Age" in df.columns


def test_imputation_mediane():
    df = pd.DataFrame({"x": [1.0, 2.0, np.nan, 4.0, 5.0]})
    df_out = impute_missing_values(df)
    assert df_out["x"].isnull().sum() == 0
    assert df_out["x"].iloc[2] == pytest.approx(3.0)


def test_suppression_si_trop_de_manquants():
    df = pd.DataFrame({
        "a": [1.0, 2.0, 3.0],
        "b": [np.nan, np.nan, np.nan],
    })
    df_out = impute_missing_values(df, deletion_threshold=40)
    assert "b" not in df_out.columns
    assert "a" in df_out.columns
