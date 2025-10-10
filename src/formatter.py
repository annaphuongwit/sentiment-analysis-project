import pandas as pd
import argparse


def clean_data(input_path, output_path):
    df = pd.read_csv(input_path)
    df = df.dropna()
    df.to_csv(output_path, index=False)
    print(f"✅ Cleaned data saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    clean_data(args.input, args.output)
