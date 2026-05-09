from pathlib import Path

from dotenv import load_dotenv
from loguru import logger

# Load environment variables from .env file
load_dotenv()

# Configures the right paths for the project, regardless of where it's run from.
# On HPC where the package is installed (not editable), set DANLONBEN_PROJ_ROOT
# to the actual project root containing the data/ directory.
import os
PROJ_ROOT = Path(os.environ["DANLONBEN_PROJ_ROOT"]) if "DANLONBEN_PROJ_ROOT" in os.environ else Path(__file__).resolve().parents[2]
logger.info(f"PROJ_ROOT path is: {PROJ_ROOT}")

DATA_DIR = PROJ_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXTERNAL_DATA_DIR = DATA_DIR / "external"

MODELS_DIR = PROJ_ROOT / "models"

REPORTS_DIR = PROJ_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

# This determines the quality of the images created from the pdf documents. Higher DPI means better quality but also larger file sizes and longer processing times.
PNG_DPI = 300

# Document registry
# PDFs go in data/raw/<sector>/<filename>
DOCUMENTS = [
    {
        "doc_id": "finance_nationalbank_2024",
        "sector": "finance",
        "title": "Danmarks Nationalbank Årsrapport 2024",
        "filename": "aarsrapport-2024.pdf",
    },
    {
        "doc_id": "finance_statens_laantagning_2023",
        "sector": "finance",
        "title": "Statens låntagning og gæld 2023",
        "filename": "statens-laantagning-og-gaeld-2023.pdf",
    },
    {
        "doc_id": "healthcare_sundhedsstyrelsen_2023",
        "sector": "Health",
        "title": "Sundhedsstyrelsen Årsrapport 2023",
        "filename": "aarsrapport-2023-for-sundhedsstyrelsen.pdf",
    },
    {
        "doc_id": "healthcare_sundhedsprofil_2023",
        "sector": "Health",
        "title": "Nationale Sundhedsprofil 2023",
        "filename": "web_sundhedsprofilen_2023_kort-a.pdf",
    },
    {
        "doc_id": "legal_rigsrevisionen_2023",
        "sector": "legal",
        "title": "Rigsrevisionen Beretning om revisionen af statens forvaltning i 2023",
        "filename": "SR1923.pdf",
    },
    {
        "doc_id": "energy_energistatistik_2023",
        "sector": "Energy",
        "title": "Energistatistik 2023",
        "filename": "energistatistik_2023.pdf",
    },
    {
        "doc_id": "energy_forsyningspolitisk_2024",
        "sector": "Energy",
        "title": "Energi- og forsyningspolitisk redegørelse 2024",
        "filename": "Energi- og forsyningspolitisk redegørelse 2024.pdf",
    },
    {
        "doc_id": "municipal_kbh_2023",
        "sector": "Municipality",
        "title": "Københavns Kommune Årsrapport 2023",
        "filename": "Københavns kommunes årsrapport 2023.pdf",
    },
]

# If tqdm is installed, configure loguru with tqdm.write
# https://github.com/Delgan/loguru/issues/135
try:
    from tqdm import tqdm

    logger.remove(0)
    logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True)
except ModuleNotFoundError:
    pass
