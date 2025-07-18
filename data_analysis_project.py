
import pandas as pd
import matplotlib.pyplot as plt

# Load Excel data
df = pd.read_excel("sales_data.xlsx")

# Display basic info
print("Data Overview:")
print(df.head())

# Summary statistics
print("\nSummary:")
print(df.describe())

# Monthly sales plot
monthly_sales = df.groupby("Month")["Sales"].sum()
monthly_sales.plot(kind="bar", title="Monthly Sales", xlabel="Month", ylabel="Sales")
plt.tight_layout()
plt.savefig("monthly_sales_plot.png")
plt.show()
