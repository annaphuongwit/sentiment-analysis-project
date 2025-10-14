import pandas as pd
import argparse


def clean_data(input_path, output_path):
    df = pd.read_csv(input_path)
    df = df.dropna()
    df.to_csv(output_path, index=False)
    print(f"✅ Loading data from {input_path}")
    print(f"Cleaning data...")
    print(f"✅ Data cleaned: {len(df)} rows")
    print(f"✅ Cleaned data INTO INTO saved to {output_path}")
    print("Done")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    clean_data(args.input, args.output)
