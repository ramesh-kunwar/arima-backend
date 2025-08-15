#!/usr/bin/env python3
"""
Data preparation script for pizza sales forecasting
Aggregates daily pizza sales data for ARIMA forecasting
"""
import pandas as pd
import numpy as np
from datetime import datetime

def prepare_pizza_sales_data():
    """
    Load and aggregate pizza sales data by date for forecasting
    """
    print("Loading pizza sales data...")
    
    # Load the data
    df = pd.read_csv('pizza-sales.csv')
    print(f"Loaded {len(df)} pizza sale records")
    
    # Display data info
    print("\nData columns:", list(df.columns))
    print("\nFirst few rows:")
    print(df.head())
    
    # Convert order_date to datetime with flexible parsing
    df['order_date'] = pd.to_datetime(df['order_date'], format='mixed', dayfirst=False)
    
    # Aggregate sales by date
    daily_sales = df.groupby('order_date').agg({
        'total_price': 'sum',           # Total revenue per day
        'quantity': 'sum',              # Total pizzas sold per day
        'pizza_id': 'count'             # Total orders per day
    }).reset_index()
    
    # Rename columns for clarity
    daily_sales.columns = ['date', 'total_revenue', 'total_quantity', 'total_orders']
    
    print(f"\nAggregated to {len(daily_sales)} daily records")
    print(f"Date range: {daily_sales['date'].min()} to {daily_sales['date'].max()}")
    
    # Show summary statistics
    print("\nDaily Sales Summary:")
    print(f"Average daily revenue: ${daily_sales['total_revenue'].mean():.2f}")
    print(f"Average daily quantity: {daily_sales['total_quantity'].mean():.1f} pizzas")
    print(f"Average daily orders: {daily_sales['total_orders'].mean():.1f} orders")
    
    # Create separate CSV files for different metrics
    metrics = ['total_revenue', 'total_quantity', 'total_orders']
    
    for metric in metrics:
        # Create a simplified CSV with just date and value for ARIMA
        forecast_data = daily_sales[['date', metric]].copy()
        forecast_data.columns = ['date', 'value']
        
        # Save to CSV
        filename = f'pizza_sales_{metric}_daily.csv'
        forecast_data.to_csv(filename, index=False)
        print(f"Created {filename} for forecasting")
    
    print("\nData preparation completed!")
    return daily_sales

if __name__ == "__main__":
    daily_sales = prepare_pizza_sales_data()
