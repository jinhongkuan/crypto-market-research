from datetime import datetime
import os
import dotenv
from pathlib import Path

import typer
from loguru import logger
from dune_client.client import DuneClient

from amm.config import RAW_DATA_DIR
from amm.dune.config import DEX_TRADES_QUERY_ID
from amm.types import TokenPair

from dune_client.types import QueryParameter, ParameterType
from dune_client.client import DuneClient 
from dune_client.query import QueryBase

app = typer.Typer()
dotenv.load_dotenv()

def filter_token_pairs(raw_token_pairs: list[str]):
    return [TokenPair(pair.split("-")[0], pair.split("-")[1]) for pair in raw_token_pairs]

# dex.trades 
@app.command()
def fetch_daily_volume(
    start_date: str,
    end_date: str,
    token_pairs: list[str],
    output_file: Path,
):
    api_key = os.getenv("DUNE_API_KEY")
    if not api_key:
        raise ValueError("No API key provided and DUNE_API_KEY not found in environment")

    query = QueryBase(
        name="dex market volume",
        query_id=DEX_TRADES_QUERY_ID,
        params=[
            QueryParameter(name="token_pairs", parameter_type=ParameterType.TEXT, value=(','.join(filter_token_pairs(token_pairs)))), 
            QueryParameter(name="start_date", parameter_type=ParameterType.DATE, value=datetime.strptime(start_date, "%Y-%m-%d")),
            QueryParameter(name="end_date", parameter_type=ParameterType.DATE, value=datetime.strptime(end_date, "%Y-%m-%d")),
        ],
    )
    
    dune = DuneClient(api_key)
    df = dune.run_query_dataframe(query)
    df.to_csv(RAW_DATA_DIR / output_file, index=False)

if __name__ == "__main__":
    app()
