import os
import dotenv
from pathlib import Path

import typer
from loguru import logger
from dune_client.client import DuneClient

from amm.config import PROCESSED_DATA_DIR

app = typer.Typer()

@app.command()
def fetch_dune_data(
    query_id: int,
    output_file: str,
    api_key: str = None
):
    """Fetch data from Dune Analytics and save to CSV.
    
    Args:
        query_id: ID of the Dune query to execute
        output_file: Name of output file to save results
        api_key: Dune API key (optional - will check env var if not provided)
    """
    if api_key is None:
        dotenv.load_dotenv()
        api_key = os.getenv("DUNE_API_KEY")
        if not api_key:
            raise ValueError("No API key provided and DUNE_API_KEY not found in environment")

    logger.info(f"Fetching data from Dune query {query_id}...")
    try:
        dune = DuneClient(api_key)
        df = dune.get_latest_result_dataframe(query_id)
        output_path = PROCESSED_DATA_DIR / output_file
        df.to_csv(output_path, index=False)
        logger.success(f"Successfully saved data to {output_path}")
    except Exception as e:
        logger.error(f"Error fetching data from Dune: {str(e)}")
        raise

if __name__ == "__main__":
    app()
