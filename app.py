
import streamlit as st
import pandas as pd

# Configure the application
st.set_page_config(
    page_title="IronLedger Analytics",
    page_icon="🚛",
    layout="wide"
)

# Load our synthetic truck sales data
df = pd.read_csv("data/truck_sales.csv")

# Dashboard title
st.title("🚛 IronLedger Analytics")
st.subheader("Commercial Truck Dealership Intelligence")

st.divider()

# Calculate business metrics
total_revenue = df["sale_price"].sum()
total_profit = df["gross_profit"].sum()
trucks_sold = len(df)
profit_margin = (total_profit / total_revenue) * 100

# Display four KPI cards
col1, col2, col3, col4 = st.columns(4)

col1.metric("Total Revenue", f"${total_revenue:,.0f}")
col2.metric("Gross Profit", f"${total_profit:,.0f}")
col3.metric("Trucks Sold", f"{trucks_sold:,}")
col4.metric("Gross Margin", f"{profit_margin:.1f}%")

st.divider()

# Analyze sales by manufacturer
st.header("Sales by Manufacturer")

manufacturer_sales = (
    df.groupby("manufacturer")["sale_price"]
    .sum()
    .sort_values(ascending=False)
)

st.bar_chart(manufacturer_sales)

# Display transaction records
st.header("Truck Sales Transactions")
st.dataframe(df, use_container_width=True)
