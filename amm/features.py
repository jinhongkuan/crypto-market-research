import os
from pathlib import Path

import typer
from loguru import logger
from tqdm import tqdm
from web3 import Web3
import pandas as pd
import numpy as np
from amm.config import PROCESSED_DATA_DIR, RAW_DATA_DIR, RPC_URLS, INTERIM_DATA_DIR, EXTERNAL_DATA_DIR
from amm.utils import parse_filename_fee_switch
from sklearn.preprocessing import MinMaxScaler

app = typer.Typer()

def get_block_timestamp(w3: Web3, block: int) -> int:
    blockData =  w3.eth.get_block(block)
    timestamp = blockData.get("timestamp")
    if timestamp is None: 
        raise Exception(f"Block {block} not found")
    return timestamp

@app.command()
def process_v3_swaps(
    input_filename: str,
    output_filename: str,
    chain: str,
    interval: str,
):
    """Process V3 swaps data and compute derived metrics.
    
    Args:
        input_path: Path to input CSV file containing raw swaps data
        output_path: Path to save processed CSV file
        chain: Chain to process - either 'ethereum' or 'arbitrum'
        interval: Aggregation interval - either 'block' or 'day'
    """
    if interval not in ["block", "day"]:
        raise ValueError("Interval must be either 'block' or 'day'")

    logger.info(f"Processing V3 swaps data from {RAW_DATA_DIR / input_filename}...")

    # Read input CSV
    df = pd.read_csv(RAW_DATA_DIR / input_filename)
    df["liquidity"] = df["liquidity"].astype(float)
    df["amount0"] = df["amount0"].astype(float)
    df["sqrtPriceX96"] = df["sqrtPriceX96"].astype(float)
    
    # Initialize web3 connection
    w3 = Web3(Web3.HTTPProvider(RPC_URLS[chain]))

    # Get block timestamps
    first_block_timestamp = get_block_timestamp(w3, int(df.iloc[0].blockNumber))
    last_block_timestamp = get_block_timestamp(w3, int(df.iloc[-1].blockNumber))
    
    # Calculate time spans
    total_time_span = last_block_timestamp - first_block_timestamp
    total_blocks = df.blockNumber.max() - df.blockNumber.min()

    # Estimate timestamps
    def estimate_timestamp(block):
        block_fraction = (block - df.blockNumber.min()) / total_blocks
        return int(first_block_timestamp + block_fraction * total_time_span)

    # Convert timestamp for day interval
    if interval == "day":
        df["time"] = df.apply(lambda row: estimate_timestamp(row.blockNumber), axis=1)
        df["time"] = pd.to_datetime(df["time"], unit='s')
        df.set_index("time", inplace=True)
    else:
        df.set_index("blockNumber", inplace=True)

    # Filter out zero amounts
    df = df[df["amount0"] != 0]

    # Process required columns
    df["liquidity"] = np.log(df["liquidity"])
    df["price"] = df["sqrtPriceX96"].astype(float).pow(2).apply(lambda x: x / 2**96)

    if interval == "day":
        # For daily interval
        # Calculate volume as log of sum of absolute amounts
        volume = df.groupby(pd.Grouper(freq='D'))['amount0'].apply(lambda x: np.log(x.abs().sum()))
        
        # Get median liquidity
        liquidity = df.groupby(pd.Grouper(freq='D'))['liquidity'].median()
        
        # Calculate OHLC for price
        price_open = df.groupby(pd.Grouper(freq='D'))['price'].first().apply(np.log)
        price_high = df.groupby(pd.Grouper(freq='D'))['price'].max().apply(np.log)
        price_low = df.groupby(pd.Grouper(freq='D'))['price'].min().apply(np.log)
        price_close = df.groupby(pd.Grouper(freq='D'))['price'].last().apply(np.log)
        
        price_ohlc = pd.DataFrame({
            'open': price_open,
            'high': price_high,
            'low': price_low,
            'close': price_close
        })
        # Combine all metrics
        df = pd.concat([volume.rename('volume'), liquidity, price_ohlc], axis=1)
        
        df = df[['volume', 'liquidity', 'open', 'high', 'low', 'close']]
        
    else:
        # For block interval
        df["amount0"] = np.log(df["amount0"].abs())
        df["price"] = np.log(df["price"])
        
        df = df[['amount0', 'liquidity', 'price']]
    # Drop NA rows
    df.dropna(inplace=True)
        
    # Save processed data
    df.to_csv(INTERIM_DATA_DIR / output_filename)
    logger.success(f"Successfully processed data and saved to {INTERIM_DATA_DIR / output_filename}")
    


@app.command()
def process_v2_swaps(
    input_filename: str = "swaps",
    output_filename: str = "processed_swaps",
    chain: str = "ethereum",
    interval: str = "day",
):
    pass


@app.command()
def process_external_market_share(
    input_filename: str,
    output_filename: str,
):
    df = pd.read_csv(EXTERNAL_DATA_DIR / input_filename)
    df['token_pair'] = df['token_pair'].str.replace('-', '_').str.lower()
    df = df[df['project'] == 'uniswap']
    df['chain'] = df['blockchain']
    df['date'] = pd.to_datetime(df['time']).dt.date
    df['volume'] = df['project_volume']
    df.set_index('date', inplace=True)
    df = df[['chain', 'token_pair', 'volume', 'market_share_percentage']]
    df.to_csv(INTERIM_DATA_DIR / output_filename)

@app.command()
def process_daily(
    directory_path: str,
    output_filename: str,
):
    aggregated_df = pd.DataFrame()

    market_share_df = pd.read_csv(INTERIM_DATA_DIR / 'features' / 'market_share.csv')

    for file in os.listdir(INTERIM_DATA_DIR / directory_path):
        df = pd.read_csv(INTERIM_DATA_DIR / directory_path / file)
        # Extract metadata from filename
        chain, token_pair, fee_tier, fee_switch = parse_filename_fee_switch(file)
        df['chain'] = chain
        df['token_pair'] = token_pair

        ms_df = market_share_df[market_share_df['token_pair'] == token_pair]
        df['market_share'] = df['time'].map(ms_df.set_index('date')['market_share_percentage'])
        df['fee_tier'] = fee_tier
        df['fee_switch'] = fee_switch == 'after'

        # Normalize per pool
        scaler = MinMaxScaler()
        df[['volume', 'liquidity', 'open', 'high', 'low', 'close']] = scaler.fit_transform(df[['volume', 'liquidity', 'open', 'high', 'low', 'close']])
        aggregated_df = pd.concat([aggregated_df, df])

    aggregated_df.to_csv(PROCESSED_DATA_DIR / output_filename, index=False)


if __name__ == "__main__":
    app()