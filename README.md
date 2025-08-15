# ARIMA Forecasting Backend

A complete Python Flask backend for ARIMA-based time series forecasting using Statsmodels.

## Features

- **Professional ARIMA Implementation**: Automatic order selection (p,d,q) using AIC criterion
- **Comprehensive API**: Upload, train, forecast, and download results
- **Statistical Analysis**: Stationarity testing, residual analysis, performance metrics
- **Model Persistence**: Save and load trained models for future use
- **Performance Metrics**: RMSE, MAE, MAPE, R², directional accuracy
- **Confidence Intervals**: Statistical confidence bounds for predictions
- **Train-Test Split**: Automatic data splitting for model validation

## Technology Stack

- **Framework**: Python + Flask + Flask-CORS
- **Time Series**: Statsmodels (ARIMA implementation)
- **Data Processing**: Pandas + NumPy
- **Machine Learning**: Scikit-learn (for metrics)
- **Model Storage**: Joblib
- **Deployment**: Gunicorn ready

## Installation

### Quick Start
```bash
cd arima-backend
./start.sh
```

### Manual Installation
```bash
# Create virtual environment
python3 -m venv arima_env
source arima_env/bin/activate

# Install dependencies
pip install -r requirements.txt

# Start the service
python app.py
```

## API Endpoints

### Health Check
- **GET** `/` - Service health check and status

### Data Upload & Training
- **POST** `/upload` - Upload CSV file and start ARIMA training

**Request Format:**
```bash
curl -X POST http://localhost:8080/upload \
  -F "file=@your_data.csv" \
  -F "date_column=date" \
  -F "value_column=value" \
  -F "test_size=0.2" \
  -F "forecast_horizon=30"
```

**Response:**
```json
{
  "session_id": "uuid-here",
  "status": "success",
  "message": "ARIMA model trained successfully",
  "data": {
    "session_id": "uuid",
    "filename": "data.csv",
    "model_info": {
      "order": [2, 1, 1],
      "aic": 1234.56,
      "bic": 1245.78,
      "stationarity_test": {...}
    },
    "performance_metrics": {
      "rmse": 0.123,
      "mae": 0.098,
      "mape": 5.2,
      "r2": 0.85
    },
    "test_forecast": {...},
    "future_forecast": {...}
  }
}
```

### Get Forecast Results
- **GET** `/forecast/{session_id}` - Retrieve forecast results

### List Training Sessions
- **GET** `/sessions` - List all training sessions

### Download Results
- **GET** `/download/{session_id}` - Download results as JSON file

### Model Retraining
- **POST** `/retrain/{session_id}` - Retrain with different parameters

## CSV File Format

Your CSV file should contain:
- **Date column**: Any standard date format (YYYY-MM-DD, MM/DD/YYYY, etc.)
- **Value column**: Numeric values for forecasting
- **Header row**: Column names in the first row

**Example:**
```csv
date,sales
2024-01-01,100.5
2024-01-02,102.3
2024-01-03,98.7
...
```

## ARIMA Model Features

### Automatic Order Selection
- Tests combinations of p (0-5), d (0-2), q (0-5)
- Uses AIC criterion for optimal model selection
- Automatic stationarity testing and differencing

### Statistical Analysis
- **Augmented Dickey-Fuller Test**: Stationarity testing
- **Residual Analysis**: Mean, std, skewness, kurtosis
- **Model Diagnostics**: AIC, BIC, Log-Likelihood

### Performance Metrics
- **RMSE**: Root Mean Square Error
- **MAE**: Mean Absolute Error  
- **MAPE**: Mean Absolute Percentage Error
- **R²**: Coefficient of Determination
- **Directional Accuracy**: Trend prediction accuracy

## Configuration

### Environment Variables
```bash
export PORT=8080              # Service port
export DEBUG=True             # Debug mode
export FLASK_ENV=development  # Flask environment
```

### Model Parameters
- **Max P**: Maximum autoregressive order (default: 5)
- **Max Q**: Maximum moving average order (default: 5)
- **Max D**: Maximum differencing order (default: 2)

## File Structure

```
arima-backend/
├── app.py                 # Main Flask application
├── arima_model.py         # ARIMA model implementation
├── requirements.txt       # Python dependencies
├── start.sh              # Startup script
├── README.md             # This file
├── uploads/              # Uploaded CSV files
├── saved_models/         # Trained model files
└── results/              # Training session results
```

## Production Deployment

### Using Gunicorn
```bash
gunicorn -w 4 -b 0.0.0.0:8080 app:app
```

### Docker Deployment
```dockerfile
FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 8080
CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:8080", "app:app"]
```

## API Usage Examples

### Python Client
```python
import requests
import json

# Upload and train
files = {'file': open('data.csv', 'rb')}
data = {
    'date_column': 'date',
    'value_column': 'sales',
    'test_size': 0.2,
    'forecast_horizon': 30
}

response = requests.post('http://localhost:8080/upload', files=files, data=data)
result = response.json()
session_id = result['session_id']

# Get forecast
forecast = requests.get(f'http://localhost:8080/forecast/{session_id}')
print(forecast.json())
```

### JavaScript/React Client
```javascript
const formData = new FormData();
formData.append('file', csvFile);
formData.append('date_column', 'date');
formData.append('value_column', 'sales');
formData.append('test_size', '0.2');
formData.append('forecast_horizon', '30');

const response = await fetch('http://localhost:8080/upload', {
  method: 'POST',
  body: formData
});

const result = await response.json();
console.log('Training result:', result);
```

## Troubleshooting

### Common Issues

**Port Already in Use**
```bash
lsof -ti:8080 | xargs kill -9
```

**Import Errors**
```bash
pip install --upgrade setuptools
pip install -r requirements.txt
```

**Permission Denied (start.sh)**
```bash
chmod +x start.sh
```

**Low Memory**
```bash
# Reduce dataset size or use fewer ARIMA parameters
# Set max_p=3, max_q=3 in arima_model.py
```

## Model Performance Tips

1. **Data Quality**: Ensure clean, consistent time series data
2. **Sample Size**: Minimum 50 data points, ideally 100+
3. **Stationarity**: The model automatically handles differencing
4. **Seasonality**: Consider seasonal decomposition for complex patterns
5. **Validation**: Use train-test split to evaluate performance

## Final Year Project Integration

This backend is designed for academic demonstration:

- **Complete Implementation**: Professional-grade ARIMA forecasting
- **Educational Value**: Clear code structure and documentation
- **Real-world Application**: Suitable for business forecasting
- **Extensible**: Easy to add new features or models
- **Well-documented**: Comprehensive API and code documentation

## Support

For issues or questions:
1. Check the logs in the console output
2. Verify CSV file format and data quality
3. Ensure backend service is running on correct port
4. Test with sample data first

## License

This project is created for educational purposes as part of a Final Year Project.
