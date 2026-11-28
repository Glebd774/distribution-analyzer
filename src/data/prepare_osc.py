import pandas as pd


df = pd.read_csv(
    "data/oscilloscope.txt", 
    sep="\t", 
    skiprows=lambda x: x < 10, 
    engine="python"
)


col_name = [c for c in df.columns if "Voltage" in c][0]
df = df[["index", col_name]].rename(columns={col_name: "voltage"})

# Сохраняем в чистый CSV
df.to_csv("data/osc_data.csv", index=False)
print(f"Готово: data/osc_data.csv ({len(df)} строк)")