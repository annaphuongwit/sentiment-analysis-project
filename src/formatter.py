import argparse
import logging

import pandas as pd

log = logging.getLogger(__name__)

def clean_data(input_path, output_path):
    df = pd.read_csv(input_path)
    df = df.dropna()
    df.to_csv(output_path, index=False)
    log.info("✅ Loading data from {input_path}")
    log.info("Cleaning data...")
    log.info("✅ Data cleaned: {len(df)} rows")
    log.info("✅ Cleaned data INTO INTO saved to {output_path}")
    log.info("Done")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    clean_data(args.input, args.output)
