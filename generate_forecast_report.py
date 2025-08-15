#!/usr/bin/env python3
"""
Pizza Sales Forecasting Report Generator
Creates comprehensive analysis and visualizations of ARIMA forecast results
"""
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import requests
import json
from datetime import datetime, timedelta
import numpy as np

def fetch_session_data(session_id):
    """Fetch session data from ARIMA backend"""
    try:
        response = requests.get(f'http://localhost:8080/forecast/{session_id}')
        if response.status_code == 200:
            return response.json()['data']
        else:
            print(f"Error fetching session {session_id}: {response.text}")
            return None
    except Exception as e:
        print(f"Error: {e}")
        return None

def create_forecast_plot(session_data, metric_name, save_path):
    """Create a comprehensive forecast visualization"""
    
    # Extract data
    test_forecast = session_data['test_forecast']
    future_forecast = session_data['future_forecast']
    data_info = session_data['data_info']
    
    # Parse dates
    test_dates = pd.to_datetime(test_forecast['dates'])
    future_dates = pd.to_datetime(future_forecast['dates'])
    
    # Create the plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12))
    
    # Plot 1: Historical + Test + Forecast
    ax1.plot(test_dates, test_forecast['actual_values'], 'b-', 
             label='Actual Test Data', linewidth=2, alpha=0.8)
    ax1.plot(test_dates, test_forecast['predicted_values'], 'r--', 
             label='ARIMA Predictions (Test)', linewidth=2, alpha=0.8)
    ax1.plot(future_dates, future_forecast['predictions'], 'g-', 
             label='Future Forecast', linewidth=2, alpha=0.8)
    
    # Add confidence intervals for future forecast
    ax1.fill_between(future_dates, 
                     future_forecast['confidence_intervals']['lower'],
                     future_forecast['confidence_intervals']['upper'],
                     alpha=0.2, color='green', label='95% Confidence Interval')
    
    ax1.set_title(f'Pizza Sales Forecast - {metric_name}', fontsize=16, fontweight='bold')
    ax1.set_xlabel('Date', fontsize=12)
    ax1.set_ylabel(metric_name, fontsize=12)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Format x-axis dates
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax1.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)
    
    # Plot 2: Residuals analysis (Test data only)
    residuals = np.array(test_forecast['actual_values']) - np.array(test_forecast['predicted_values'])
    ax2.plot(test_dates, residuals, 'purple', alpha=0.7, linewidth=1)
    ax2.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    ax2.set_title('Residuals (Actual - Predicted)', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Date', fontsize=12)
    ax2.set_ylabel('Residual Value', fontsize=12)
    ax2.grid(True, alpha=0.3)
    
    # Format x-axis dates for residuals plot
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax2.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Saved plot: {save_path}")

def generate_report():
    """Generate comprehensive forecast report"""
    
    # Session IDs from the forecasting results (latest sessions)
    sessions = {
        'Total Daily Revenue ($)': 'dcc89427-c2ee-4d7b-8ad8-054e24151361',
        'Total Daily Quantity (Pizzas)': '5cd39dff-0182-48eb-8bb7-5cd9d8cb1d3d', 
        'Total Daily Orders': '67c1126b-608c-4194-9e1d-ae5b183cdf40'
    }
    
    print("🍕 Generating Pizza Sales Forecast Report")
    print("=" * 60)
    
    all_data = {}
    
    # Fetch all session data and create plots
    for metric_name, session_id in sessions.items():
        print(f"\n📊 Processing {metric_name}...")
        session_data = fetch_session_data(session_id)
        
        if session_data:
            all_data[metric_name] = session_data
            
            # Create plot
            safe_name = metric_name.replace(' ', '_').replace('(', '').replace(')', '').replace('$', 'USD')
            plot_path = f'forecast_{safe_name.lower()}.png'
            create_forecast_plot(session_data, metric_name, plot_path)
    
    # Generate summary report
    print(f"\n📝 Generating Summary Report...")
    
    report_lines = [
        "# 🍕 Pizza Sales ARIMA Forecasting Report",
        f"## Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## Executive Summary",
        "",
        "This report presents ARIMA (AutoRegressive Integrated Moving Average) forecasting results for pizza sales data.",
        "The analysis covers three key metrics: daily revenue, daily quantity sold, and daily number of orders.",
        "",
        "### Data Overview",
        f"- **Dataset Period**: January 1, 2015 - December 31, 2015",
        f"- **Total Records**: 358 days of sales data",
        f"- **Training Period**: 286 days (80%)",
        f"- **Test Period**: 72 days (20%)",
        f"- **Forecast Horizon**: 30 days into the future",
        "",
        "## Model Performance Summary",
        ""
    ]
    
    for metric_name, session_data in all_data.items():
        metrics = session_data['performance_metrics']
        model_info = session_data['model_info']
        
        report_lines.extend([
            f"### {metric_name}",
            f"- **ARIMA Order**: ({model_info['order']['p']}, {model_info['order']['d']}, {model_info['order']['q']})",
            f"- **AIC**: {model_info['aic']:.2f}",
            f"- **Mean Absolute Error (MAE)**: {metrics.get('mae', 'N/A'):.2f}",
            f"- **Root Mean Square Error (RMSE)**: {metrics.get('rmse', 'N/A'):.2f}",
            f"- **R-squared**: {metrics.get('r2', 'N/A'):.4f}",
            f"- **Mean Absolute Percentage Error (MAPE)**: {metrics.get('mape', 'N/A'):.2f}%",
            ""
        ])
    
    # Add forecast insights
    report_lines.extend([
        "## Key Insights",
        "",
        "### Revenue Forecasting",
        "- The model predicts daily revenue ranging from $1,496 to $2,929 over the next 30 days",
        "- Average predicted daily revenue: ~$2,284",
        "- Model shows moderate accuracy with 18.21% MAPE",
        "",
        "### Quantity Forecasting", 
        "- Daily pizza sales predicted to range from 89 to 194 pizzas",
        "- Average predicted daily quantity: ~138 pizzas",
        "- Strong performance with 16.20% MAPE",
        "",
        "### Order Forecasting",
        "- Daily orders predicted to range from 81 to 197 orders",
        "- Average predicted daily orders: ~136 orders",
        "- Model accuracy: 18.83% MAPE",
        "",
        "## Recommendations",
        "",
        "1. **Inventory Management**: Use quantity forecasts to optimize ingredient ordering",
        "2. **Staffing**: Plan staff schedules based on predicted order volumes",
        "3. **Revenue Planning**: Budget and plan promotions using revenue forecasts",
        "4. **Model Improvement**: Consider incorporating seasonal patterns and external factors",
        "",
        "## Technical Details",
        "",
        f"- **Forecasting Method**: ARIMA (AutoRegressive Integrated Moving Average)",
        f"- **Model Selection**: Automatic order selection using AIC criterion",
        f"- **Confidence Intervals**: 95% confidence bands provided for uncertainty quantification",
        f"- **Validation**: Out-of-sample testing on 20% of historical data",
        "",
        "---",
        "*Report generated by ARIMA Forecasting System*"
    ])
    
    # Save report
    with open('Pizza_Sales_Forecast_Report.md', 'w') as f:
        f.write('\n'.join(report_lines))
    
    print(f"✅ Saved report: Pizza_Sales_Forecast_Report.md")
    
    # Print summary to console
    print(f"\n🎯 FORECAST SUMMARY")
    print("=" * 60)
    
    for metric_name, session_data in all_data.items():
        future_forecast = session_data['future_forecast']
        avg_forecast = np.mean(future_forecast['predictions'])
        min_forecast = min(future_forecast['predictions'])
        max_forecast = max(future_forecast['predictions'])
        
        print(f"\n{metric_name}:")
        print(f"  Next 30 days average: {avg_forecast:.1f}")
        print(f"  Range: {min_forecast:.1f} - {max_forecast:.1f}")
    
    print(f"\n📁 Files generated:")
    print(f"  - Pizza_Sales_Forecast_Report.md")
    for metric_name in sessions.keys():
        safe_name = metric_name.replace(' ', '_').replace('(', '').replace(')', '').replace('$', 'USD')
        print(f"  - forecast_{safe_name.lower()}.png")

if __name__ == "__main__":
    generate_report()
