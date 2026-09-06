import logging
from pathlib import Path


def setup_logging(
    log_dir: str | Path = "logs", log_file: str = "smart_imputer.log"
) -> None:
    """Configure le logging de l'application (console + fichier).

    Args:
        log_dir: Dossier où sera écrit le fichier de log (créé si besoin).
        log_file: Nom du fichier de log.
    """
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    log_format = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"

    logging.basicConfig(
        level=logging.INFO,
        format=log_format,
        handlers=[
            logging.StreamHandler(),  # affichage dans la console
            logging.FileHandler(
                log_dir / log_file, mode="a", encoding="utf-8"
            ),  # écriture dans le fichier
        ],
    )
