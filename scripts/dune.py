import os
import dotenv
import argparse
from dune_client.client import DuneClient

dotenv.load_dotenv()

parser = argparse.ArgumentParser(description='Fetch data from Dune Analytics')
parser.add_argument('query_id', type=int)
parser.add_argument('output_file', type=str)

api_key = os.getenv("DUNE_API_KEY")
args = parser.parse_args()


dune = DuneClient(api_key)
df = dune.get_latest_result_dataframe(args.query_id)
df.to_csv(f"data/external/{args.output_file}", index=False)