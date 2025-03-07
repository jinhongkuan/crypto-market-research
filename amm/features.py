

import typer
import pandas as pd 

app = typer.Typer()

# Processors take in raw data and return processed data 

class MarketShareProcessor:
    def __init__(self, df: pd.DataFrame):
        df['volume_usd'] = pd.to_numeric(df['volume_usd'], errors='coerce') 
        df.rename(columns={"block_date": "date"}, inplace=True)
        df = df.dropna()
        self.df = df

    def rank_entities(self):
        return self.df.groupby(["project", "version"]).agg({"volume_usd": "sum"}) \
                   .sort_values(by="volume_usd", ascending=False).index.tolist()
    
    def feature_volume_by_token_pair(self):
        groups = self.df.groupby("token_pair")
    
        def iterator():
            for token_pair, group in groups:
                group = group.groupby(["date", "project", "version"]).agg({"volume_usd": "sum"}).reset_index().fillna(0)
                project_volume = group.pivot(
                    index='date',
                    columns=['project', 'version'],
                    values='volume_usd'
                )
                yield token_pair, project_volume

        return {
            "count": groups.ngroups,
            "iterator": iterator
        }
    
    def feature_volume_by_blockchain(self, top_k: int = 100):
        groups = self.df.groupby("blockchain")
        selected = groups.agg({"volume_usd": "sum"}).sort_values(by="volume_usd", ascending=False).head(top_k)
    
        def iterator():
            for blockchain, group in groups:
                if blockchain not in selected.index:
                    continue

                group = group.groupby(["date", "project", "version"]).agg({"volume_usd": "sum"}).reset_index().fillna(0)
                project_volume = group.pivot(
                    index='date',
                    columns=['project', 'version'],
                    values='volume_usd'
                )
                yield blockchain, project_volume

        return {
            "count": len(selected),
            "iterator": iterator
        }

        
if __name__ == "__main__":
    app()