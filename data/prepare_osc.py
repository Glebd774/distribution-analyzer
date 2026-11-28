import pandas as pd

df = pd.read_csv("data/oscilloscope.txt", sep="\t", skiprows=8)

df = df.rename(columns={"CH1_Voltage(mV)": "voltage"})

# Сохраняем в чистый CSV
df.to_csv("data/osc_data.csv", index=False)
print(f"Готово: data/osc_data.csv ({len(df)} строк)")