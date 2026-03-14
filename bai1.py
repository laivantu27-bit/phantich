import pandas as pd
import matplotlib.pyplot as plt

def load_data():
    return pd.read_csv("data.csv")

def explore(df):
    print("Dataset:")
    print(df)
    print("Shape:", df.shape)
    print(df.describe())
    print("Missing values:")
    print(df.isnull().sum())

def handle_missing(df):
    df["Price"] = df["Price"].fillna(df["Price"].mean())
    df["StockQuantity"] = df["StockQuantity"].fillna(df["StockQuantity"].median())
    return df

def clean_data(df):
    df = df[df["Price"] > 0]
    df = df[df["StockQuantity"] >= 0]
    df = df[(df["Rating"] >= 0) & (df["Rating"] <= 5)]
    return df

def smooth_price(df):
    df["Price_smooth"] = df["Price"].rolling(3).mean()

    plt.figure()
    plt.plot(df["Price"], label="Original Price")
    plt.plot(df["Price_smooth"], label="Smoothed Price")
    plt.title("Price Moving Average")
    plt.legend()
    plt.show()

def normalize(df):
    df["Category"] = df["Category"].str.lower()
    df["Description"] = df["Description"].str.strip()
    df["Price_VND"] = df["Price"] * 24000
    print("Final dataset:")
    print(df)

def main():
    df = load_data()
    explore(df)
    df = handle_missing(df)
    df = clean_data(df)
    smooth_price(df)
    normalize(df)
    print(df)

if __name__ == "__main__":
    main()