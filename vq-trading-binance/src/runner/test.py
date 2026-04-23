import pandas as pd

df = pd.read_csv("dataset_master.csv")

print("===== HEAD =====")
print(df.head())

print("\n===== INFO =====")
print(df.info())

print("\n===== DESCRIBE =====")
print(df.describe())

print("\n===== NULL =====")
print(df.isnull().sum())

print("\n===== COLUMNS =====")
print(df.columns.tolist())

print("\n===== LABEL CHECK =====")
df["future_return"] = df["close"].shift(-5) / df["close"] - 1

print(df["future_return"].describe())
print("\nPercent > 0.002:", (df["future_return"] > 0.002).mean())
print("Percent < -0.002:", (df["future_return"] < -0.002).mean())