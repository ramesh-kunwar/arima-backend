#!/usr/bin/env python3
"""
Script to upload data and run ARIMA forecasting
"""
import requests
import json
import pandas as pd

def upload_and_forecast(filename, metric_name):
    """
    Upload CSV file and get ARIMA forecast
    """
    print(f"\n=== Forecasting {metric_name} ===")
    
    # Prepare the file and form data
    files = {'file': open(filename, 'rb')}
    data = {
        'date_column': 'date',
        'value_column': 'value',
        'test_size': '0.2',
        'forecast_horizon': '30'
    }
    
    try:
        # Upload file and start training
        print(f"Uploading {filename}...")
        response = requests.post('http://localhost:8080/upload', files=files, data=data)
        
        if response.status_code == 200:
            result = response.json()
            session_id = result['session_id']
            print(f"✅ Upload successful! Session ID: {session_id}")
            
            # Print summary information
            data_info = result['data']['data_info']
            model_info = result['data']['model_info']
            metrics = result['data']['performance_metrics']
            
            print(f"\n📊 Data Summary:")
            print(f"   - Total records: {data_info['total_records']}")
            print(f"   - Training size: {data_info['train_size']}")
            print(f"   - Test size: {data_info['test_size']}")
            print(f"   - Date range: {data_info['date_range']['start']} to {data_info['date_range']['end']}")
            
            print(f"\n🤖 Model Info:")
            print(f"   - ARIMA Order: ({model_info['order']['p']}, {model_info['order']['d']}, {model_info['order']['q']})")
            print(f"   - AIC: {model_info['aic']:.2f}")
            print(f"   - BIC: {model_info['bic']:.2f}")
            
            print(f"\n📈 Performance Metrics:")
            print(f"   - MAE: {metrics.get('mae', 'N/A'):.2f}")
            print(f"   - RMSE: {metrics.get('rmse', 'N/A'):.2f}")
            print(f"   - R²: {metrics.get('r2', 'N/A'):.4f}")
            if metrics.get('mape') is not None:
                print(f"   - MAPE: {metrics['mape']:.2f}%")
            
            # Get future forecasts
            future_forecast = result['data']['future_forecast']
            print(f"\n🔮 Future Forecast (Next 30 days):")
            
            for i, (date, pred, lower, upper) in enumerate(zip(
                future_forecast['dates'][:10],  # Show first 10 days
                future_forecast['predictions'][:10],
                future_forecast['confidence_intervals']['lower'][:10],
                future_forecast['confidence_intervals']['upper'][:10]
            )):
                print(f"   {date[:10]}: {pred:.2f} (CI: {lower:.2f} - {upper:.2f})")
            
            if len(future_forecast['dates']) > 10:
                print(f"   ... and {len(future_forecast['dates']) - 10} more days")
            
            print(f"\n💾 Session saved as: {session_id}")
            return session_id, result
            
        else:
            print(f"❌ Upload failed: {response.status_code}")
            print(f"Error: {response.text}")
            return None, None
            
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return None, None
    finally:
        files['file'].close()

def main():
    """
    Main function to forecast different metrics
    """
    print("🍕 Pizza Sales ARIMA Forecasting")
    print("=" * 50)
    
    # Define the files to forecast
    forecast_files = [
        ('pizza_sales_total_revenue_daily.csv', 'Total Daily Revenue'),
        ('pizza_sales_total_quantity_daily.csv', 'Total Daily Quantity'),
        ('pizza_sales_total_orders_daily.csv', 'Total Daily Orders')
    ]
    
    results = {}
    
    for filename, metric_name in forecast_files:
        session_id, result = upload_and_forecast(filename, metric_name)
        if session_id:
            results[metric_name] = {
                'session_id': session_id,
                'result': result
            }
    
    print(f"\n🎯 Summary of Results:")
    print("=" * 50)
    for metric, data in results.items():
        print(f"✅ {metric}: Session {data['session_id']}")
    
    print(f"\n🌐 Access results at: http://localhost:8080/sessions")

if __name__ == "__main__":
    main()
