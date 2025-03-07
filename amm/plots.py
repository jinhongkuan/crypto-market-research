from pathlib import Path
from typing import Annotated

import typer
from loguru import logger
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
from typing import Generator

from amm.config import FIGURES_DIR, PROCESSED_DATA_DIR
from amm.features import MarketShareProcessor

app = typer.Typer()


@app.command()
def plot_market_share_token_pair(
        input_path: Path,
        output_path: Path,  
        top_k: int = typer.Option(5, help="Number of top venues to include"),
        drop_k: int = typer.Option(0, help="Number of top venues to exclude"),
        normalize: bool = typer.Option(False, help="Whether to normalize volumes")
):
    df = pd.read_csv(input_path)
    
    # Data processing
    market_share_processor = MarketShareProcessor(df)
    feature = market_share_processor.feature_volume_by_token_pair()
    num_token_pairs = feature["count"]
    volume_by_token_pair = feature["iterator"]
    venues_by_total_traffic = market_share_processor.rank_entities()

    # Plotting
    fig = plot_longitudinal_features_of_entitites(venues_by_total_traffic[drop_k:top_k], volume_by_token_pair, num_token_pairs, normalize, {"xlabel": "Date", "ylabel": "Volume (USD)"})
    fig.savefig(FIGURES_DIR / output_path)

@app.command()
def plot_market_share_blockchain(
        input_path: Path,
        output_path: Path,  
        top_k: int = typer.Option(5, help="Number of top venues to include"),
        drop_k: int = typer.Option(0, help="Number of top venues to exclude"),
        num_chains: int = typer.Option(100, help="Number of chains to include"),
        normalize: bool = typer.Option(False, help="Whether to normalize volumes")
):
    df = pd.read_csv(input_path)
    
    # Data processing
    market_share_processor = MarketShareProcessor(df)
    feature = market_share_processor.feature_volume_by_blockchain(num_chains)
    num_token_pairs = feature["count"]
    volume_by_token_pair = feature["iterator"]
    venues_by_total_traffic = market_share_processor.rank_entities()

    # Plotting
    fig = plot_longitudinal_features_of_entitites(venues_by_total_traffic[drop_k:top_k], volume_by_token_pair, num_token_pairs, normalize, {"xlabel": "Date", "ylabel": "Volume (USD)"})
    fig.savefig(FIGURES_DIR / output_path)

def plot_longitudinal_features_of_entitites(sorted_entities: list, feature_iterator, feature_cardinality: int, normalize: bool, pyplot_kwargs: dict):
    """
    Create stacked area plots with shared legend, for longitudinal data

    Args:
        feature_iterator: Iterator of tuples (subplot_category, DataFrame), where DataFrame has date index and entity columns
        entities: list of entities to plot
        normalize: bool, whether to normalize the data
    """

    # Setup figure
    fig, axes = plt.subplots(nrows=feature_cardinality, figsize=(12, 4*feature_cardinality), sharex=True)

    # Setup colors
    cmap = plt.cm.get_cmap(pyplot_kwargs.get("cmap", "summer"), len(sorted_entities))
    colors = {tuple(venue): cmap(i) for i, venue in enumerate(sorted_entities)}

    all_handles = []
    all_labels = []
    
    # Plot each group
    for idx, (feature_value, group) in enumerate(feature_iterator()):
        ax = axes[idx]

        columns = group.columns.intersection(sorted_entities)
        daily_sum = group[columns].sum(axis=1)

        handles = ax.stackplot(
            group.index, 
            *[group[x] if not normalize else group[x] / daily_sum for x in columns],
            colors=[colors[x] for x in columns]
        )

        for i, handle in enumerate(handles):
            label = columns[i] 
            if label not in all_labels:
                all_handles.append(handle)
                all_labels.append(label)

        ax.set_title(feature_value)

 
    
    # Sort and add legend
    (all_handles, all_labels) = zip(*sorted(zip(all_handles, all_labels), key=lambda x: sorted_entities.index(x[1])))
    plt.legend(all_handles, all_labels, loc='lower center', bbox_to_anchor=(0.5, -0.5), ncol=min(len(all_labels), 4))
    plt.subplots_adjust(bottom=-0.5)
    # Format x-axis
    dates = group.index.tolist()
    tick_indices = range(0, len(dates), len(dates)//10)
    fig.gca().set_xticks(tick_indices, [dates[i] for i in tick_indices], rotation=45)
    fig.tight_layout()
    fig.gca().set_xlabel(pyplot_kwargs.get("xlabel", "Date"))
    fig.gca().set_ylabel(pyplot_kwargs.get("ylabel", ""))

    return fig

if __name__ == "__main__":
    app()
