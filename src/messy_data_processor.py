import pandas as pd


def process_data(df: pd.DataFrame) -> pd.DataFrame:
     # we want to inspect the dataframe head during debugging, but it is unused
   data_snapshot = df.head() # noqa: F841


    # clean columns names by stripping whitespace and converting to lower case
    df.columns = df.columns.str.strip().str.lower()

    # Ensure 'price' is a numeric column, coercing errors
    if "price" in df.columns:
        df["price"] = pd.to_numeric(df["price"], errors="coerce")
    
    return df


if __name__ == "__main__":
    data = {
        "Product Name": ["  Laptop  ", "Mouse ", "Keyboard"],
        "Price": ["1200", "25.50", "75"],
    }
    initial_df = pd.DataFrame(data)

    cleaned_df = pd.DataFrame(data)

    cleaned_df = process_data(initial_df)
    print("Cleaned DataFrame:")
    print(cleaned_df)
