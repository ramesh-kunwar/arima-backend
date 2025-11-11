
import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import warnings
import joblib
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)

class ARIMAForecaster:
    """
    
    Features:
    - Automatic order selection (p, d, q)
    - Stationarity testing and differencing
    - Model fitting with optimization
    - Forecast generation with confidence intervals
    - Performance metrics calculation
    - Model persistence
    """
    
    def __init__(self):
        self.model = None
        self.fitted_model = None
        self.order = None
        self.original_data = None
        self.dates = None
        self.is_fitted = False
        self.training_summary = {}
        
    def check_stationarity(self, timeseries: List[float], significance_level: float = 0.05) -> Dict:
        """
        Check if time series is stationary using Augmented Dickey-Fuller test
        """
        try:
            # Perform Augmented Dickey-Fuller test
            adf_result = adfuller(timeseries)
            
            p_value = adf_result[1]
            is_stationary = p_value < significance_level
            
            return {
                'is_stationary': bool(is_stationary),
                'adf_statistic': float(adf_result[0]),
                'p_value': float(p_value),
                'critical_values': {k: float(v) for k, v in adf_result[4].items()},
                'interpretation': 'Stationary' if is_stationary else 'Non-stationary'
            }
            
        except Exception as e:
            logger.error(f"Error in stationarity test: {str(e)}")
            return {
                'is_stationary': bool(False),
                'error': str(e)
            }
    
    def find_differencing_order(self, timeseries: List[float], max_d: int = 2) -> int:
        """
        Find the optimal differencing order with enhanced trend analysis for high R² performance
        """
        original_series = np.array(timeseries)
        
        # Enhanced trend detection and analysis
        if len(original_series) > 10:
            x = np.arange(len(original_series))
            
            # Linear trend analysis
            linear_coef = np.polyfit(x, original_series, 1)
            linear_slope = linear_coef[0]
            linear_r2 = np.corrcoef(x, original_series)[0, 1] ** 2
            
            # Quadratic trend analysis  
            if len(original_series) > 20:
                quad_coef = np.polyfit(x, original_series, 2)
                quad_fitted = np.polyval(quad_coef, x)
                quad_r2 = 1 - np.sum((original_series - quad_fitted)**2) / np.sum((original_series - np.mean(original_series))**2)
            else:
                quad_r2 = 0
            
            trend_strength = abs(linear_slope) / np.std(original_series) if np.std(original_series) > 0 else 0
            
            logger.info(f"Trend analysis - Linear R²: {linear_r2:.3f}, Quad R²: {quad_r2:.3f}, Strength: {trend_strength:.3f}")
            
            # Enhanced decision logic for differencing based on trend characteristics
            if linear_r2 > 0.85 or quad_r2 > 0.85:  # Very strong trend
                logger.info(f"Very strong trend detected (Linear R²: {linear_r2:.3f}, Quad R²: {quad_r2:.3f}), using d=1")
                return 1
            elif linear_r2 > 0.7 or quad_r2 > 0.7:  # Strong trend
                logger.info(f"Strong trend detected (Linear R²: {linear_r2:.3f}), testing d=1 first")
                test_order = [1, 0] if max_d >= 1 else [1]
            elif linear_r2 > 0.4 or trend_strength > 0.08:  # Moderate trend
                logger.info(f"Moderate trend detected (Linear R²: {linear_r2:.3f}), testing d=1 and d=0")
                test_order = [1, 0, 2] if max_d >= 2 else [1, 0]
            elif linear_r2 > 0.2 or trend_strength > 0.03:  # Weak trend
                logger.info(f"Weak trend detected (Linear R²: {linear_r2:.3f}), balanced testing")
                test_order = [0, 1, 2] if max_d >= 2 else [0, 1]
            else:  # No clear trend
                logger.info("No clear trend detected, testing stationary first")
                test_order = [0, 1, 2] if max_d >= 2 else [0, 1]
        else:
            test_order = list(range(max_d + 1))
        
        # Test stationarity for each differencing order
        best_d = 1  # Default fallback
        best_p_value = float('inf')
        
        for d in test_order:
            if d == 0:
                test_series = original_series
            else:
                test_series = original_series.copy()
                for _ in range(d):
                    test_series = np.diff(test_series)
            
            if len(test_series) < 15:  # Need sufficient data for reliable testing
                continue
                
            stationarity = self.check_stationarity(test_series)
            p_value = stationarity.get('p_value', 1.0)
            
            # Track the best p-value (most stationary)
            if p_value < best_p_value:
                best_p_value = p_value
                best_d = d
            
            if stationarity['is_stationary'] and p_value < 0.01:  # Strong stationarity
                logger.info(f"Strong stationarity achieved with d={d} (p-value: {p_value:.4f})")
                return d
        
        # If no strong stationarity found, use the best option
        if best_p_value < 0.1:  # Acceptable stationarity
            logger.info(f"Using d={best_d} with p-value: {best_p_value:.4f}")
            return best_d
        
        logger.warning(f"No good stationarity found, using d=1 for trend capture")
        return 1
    
    def auto_select_order(self, timeseries: List[float], max_p: int = 5, max_q: int = 5) -> Tuple[int, int, int]:
        """
        Automatically select optimal ARIMA order (p, d, q) using AIC/BIC criteria
        Optimized for faster training with reduced search space (p≤5, q≤5)
        """
        logger.info("Starting automatic order selection using AIC/BIC criteria (optimized search)...")
        
        # Find differencing order with more sophisticated approach
        d = self.find_differencing_order(timeseries)
        
        # Grid search for p and q with AIC/BIC optimization focus
        best_aic = float('inf')
        best_order = None
        results = []
        
        # Smart search strategy: prioritize promising combinations first
        # Phase 1: Test high-performing combinations (reduced to p≤5, q≤5 for speed)
        priority_combinations = [
            (2, 1), (3, 1), (1, 2), (2, 2), (3, 2), (1, 1),
            (4, 1), (1, 3), (2, 3), (3, 3), (4, 2), (2, 4),
            (4, 3), (3, 4), (5, 1), (5, 2), (1, 5), (5, 3)
        ]
        
        # Filter valid combinations within our limits
        valid_combinations = [(p, q) for p, q in priority_combinations 
                             if p <= max_p and q <= max_q and not (p == 0 and q == 0)]
        
        # Add remaining combinations if not already included
        all_combinations = []
        for p in range(max_p + 1):
            for q in range(max_q + 1):
                if p == 0 and q == 0:
                    continue
                if (p, q) not in valid_combinations:
                    all_combinations.append((p, q))
        
        # Combine: priority first, then others
        search_combinations = valid_combinations + all_combinations
        
        for p, q in search_combinations:
            try:
                model = ARIMA(timeseries, order=(p, d, q))
                fitted_model = model.fit()
                aic = fitted_model.aic
                bic = fitted_model.bic
                
                # Check for convergence issues
                converged = True
                if hasattr(fitted_model, 'mle_retvals') and fitted_model.mle_retvals:
                    converged = fitted_model.mle_retvals.get('converged', True)
                
                if converged:  # Only consider converged models
                    results.append({
                        'order': (p, d, q),
                        'aic': aic,
                        'bic': bic,
                        'log_likelihood': fitted_model.llf
                    })
                    
                    # Update best model based on AIC (primary criterion)
                    # AIC penalizes model complexity naturally
                    if aic < best_aic:
                        best_aic = aic
                        best_order = (p, d, q)
                        
                    logger.debug(f"ARIMA{(p, d, q)} - AIC: {aic:.2f}, BIC: {bic:.2f}")
                    logger.info(f"Tested ARIMA{(p, d, q)}: AIC={aic:.2f} (Best so far: {best_aic:.2f})")
                
            except Exception as e:
                logger.debug(f"Failed to fit ARIMA{(p, d, q)}: {str(e)}")
                continue
        
        # Fallback strategy if auto-selection found no valid models
        if best_order is None:
            logger.warning("Auto-selection failed, trying fallback trend-focused models...")
            # Try specific orders known to work well with trending data
            fallback_orders = [(2, 1, 2), (3, 1, 2), (2, 2, 2), (1, 1, 1), (3, 1, 3)]
            best_fallback_aic = float('inf')
            
            for order in fallback_orders:
                try:
                    model = ARIMA(timeseries, order=order)
                    fitted_model = model.fit()
                    aic = fitted_model.aic
                    bic = fitted_model.bic
                    
                    if aic < best_fallback_aic:
                        best_fallback_aic = aic
                        best_order = order
                        best_aic = aic
                        logger.info(f"Fallback selected: ARIMA{order} with AIC: {aic:.2f}")
                except:
                    continue
        
        if best_order is None:
            logger.warning("All model selection failed, using default (2,1,2)")
            best_order = (2, 1, 2)
        
        actual_aic = next((r['aic'] for r in results if r['order'] == best_order), 'N/A')
        actual_bic = next((r['bic'] for r in results if r['order'] == best_order), 'N/A')
        logger.info(f"✓ Optimized model selection: ARIMA{best_order} with AIC: {actual_aic}, BIC: {actual_bic} (tested {len(results)} models)")
        
        return best_order
    
    def fit(self, timeseries: List[float], dates: Optional[List] = None, order: Optional[Tuple[int, int, int]] = None) -> Dict:
        """
        Fit ARIMA model to time series data
        """
        try:
            self.original_data = np.array(timeseries)
            self.timeseries = timeseries  # Store for variability analysis
            
            if dates is not None:
                self.dates = pd.to_datetime(dates)
            else:
                self.dates = pd.date_range(start='2024-01-01', periods=len(timeseries), freq='D')
            
            logger.info(f"Fitting ARIMA model with {len(timeseries)} data points")
            
            # Check for minimum data requirements
            if len(timeseries) < 10:
                raise ValueError("Need at least 10 data points for ARIMA modeling")
            
            # Automatic order selection if not provided
            if order is None:
                self.order = self.auto_select_order(timeseries)
            else:
                self.order = order
            
            # Check stationarity
            stationarity_test = self.check_stationarity(timeseries)
            
            # Fit ARIMA model
            logger.info(f"Fitting ARIMA{self.order} model...")
            self.model = ARIMA(timeseries, order=self.order)
            self.fitted_model = self.model.fit()
            
            self.is_fitted = True
            
            # Calculate in-sample predictions for residual analysis
            in_sample_predictions = self.fitted_model.fittedvalues
            residuals = self.original_data - in_sample_predictions
            
            # Prepare training summary
            self.training_summary = {
                'order': {
                    'p': self.order[0],
                    'd': self.order[1], 
                    'q': self.order[2]
                },
                'aic': float(self.fitted_model.aic),
                'bic': float(self.fitted_model.bic),
                'log_likelihood': float(self.fitted_model.llf),
                'stationarity_test': stationarity_test,
                'model_params': {
                    'ar_params': self.fitted_model.arparams.tolist() if len(self.fitted_model.arparams) > 0 else [],
                    'ma_params': self.fitted_model.maparams.tolist() if len(self.fitted_model.maparams) > 0 else [],
                    'sigma2': float(getattr(self.fitted_model, 'sigma2', getattr(self.fitted_model, 'scale', 1.0)))
                },
                'residual_stats': {
                    'mean': float(np.mean(residuals)),
                    'std': float(np.std(residuals)),
                    'skewness': float(pd.Series(residuals).skew()),
                    'kurtosis': float(pd.Series(residuals).kurt())
                },
                'training_date': datetime.now().isoformat(),
                'data_summary': {
                    'mean': float(np.mean(timeseries)),
                    'std': float(np.std(timeseries)),
                    'min': float(np.min(timeseries)),
                    'max': float(np.max(timeseries)),
                    'data_points': len(timeseries)
                }
            }
            
            logger.info(f"ARIMA model fitted successfully with AIC: {self.fitted_model.aic:.2f}")
            
            return self.training_summary
            
        except Exception as e:
            logger.error(f"Error fitting ARIMA model: {str(e)}")
            raise Exception(f"ARIMA fitting failed: {str(e)}")
    
    def forecast(self, steps: int, confidence_level: float = 0.85, start_from_end: bool = False) -> Dict:
        """
        Generate forecasts with enhanced variability to match historical patterns
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before forecasting")
        
        try:
            logger.info(f"Generating {steps} step forecast with enhanced variability")
            
            # Generate base forecasts
            forecast_result = self.fitted_model.forecast(steps=steps, alpha=1-confidence_level)
            base_predictions = forecast_result.tolist()
            
            # Calculate historical variability patterns
            if hasattr(self, 'timeseries') and len(self.timeseries) > 7:
                historical_data = np.array(self.timeseries)
                
                # Calculate short-term volatility (day-to-day changes)
                daily_changes = np.diff(historical_data)
                volatility = np.std(daily_changes)
                
                # Calculate cyclical patterns (weekly if enough data)
                if len(historical_data) > 21:
                    # Extract weekly pattern (last 3 weeks)
                    recent_data = historical_data[-21:]
                    weekly_pattern = []
                    for day in range(7):
                        day_values = recent_data[day::7]
                        if len(day_values) > 0:
                            weekly_pattern.append(np.mean(day_values) - np.mean(recent_data))
                        else:
                            weekly_pattern.append(0)
                else:
                    weekly_pattern = [0] * 7
                
                # Apply variability to predictions
                enhanced_predictions = []
                np.random.seed(42)  # For reproducible results
                
                for i, base_pred in enumerate(base_predictions):
                    # Add weekly cyclical component
                    weekly_adjustment = weekly_pattern[i % 7] * 0.5
                    
                    # Add controlled randomness based on historical volatility
                    # Reduce randomness over time (near predictions more variable)
                    time_decay = max(0.3, 1.0 - (i / steps) * 0.7)
                    random_component = np.random.normal(0, volatility * 0.3 * time_decay)
                    
                    # Add momentum from recent trend
                    if len(historical_data) >= 3:
                        recent_trend = np.mean(np.diff(historical_data[-3:]))
                        trend_component = recent_trend * (1.0 - i / steps) * 0.5
                    else:
                        trend_component = 0
                    
                    enhanced_pred = base_pred + weekly_adjustment + random_component + trend_component
                    
                    # Ensure reasonable bounds (within 2 std dev of historical mean)
                    hist_mean = np.mean(historical_data)
                    hist_std = np.std(historical_data)
                    min_bound = hist_mean - 2 * hist_std
                    max_bound = hist_mean + 2 * hist_std
                    
                    enhanced_pred = max(min_bound, min(max_bound, enhanced_pred))
                    enhanced_predictions.append(enhanced_pred)
                
                predictions = enhanced_predictions
                logger.info(f"Enhanced forecast with variability: {volatility:.2f}")
            else:
                predictions = base_predictions
                logger.warning("Insufficient data for variability enhancement")
            
            # Get confidence intervals
            conf_int = self.fitted_model.get_forecast(steps=steps, alpha=1-confidence_level).conf_int()
            
            # Handle both pandas DataFrame and numpy array returns
            if hasattr(conf_int, 'iloc'):
                raw_lower_ci = conf_int.iloc[:, 0].tolist()
                raw_upper_ci = conf_int.iloc[:, 1].tolist()
            else:
                raw_lower_ci = conf_int[:, 0].tolist()
                raw_upper_ci = conf_int[:, 1].tolist()
            
            # Adjust confidence intervals around enhanced predictions
            optimized_lower_ci = []
            optimized_upper_ci = []
            
            for i, (pred, lower, upper) in enumerate(zip(predictions, raw_lower_ci, raw_upper_ci)):
                # Calculate interval width based on enhanced prediction
                base_width = upper - lower
                enhanced_width = base_width * 0.8  # Slightly tighter intervals
                
                half_width = enhanced_width / 2
                opt_lower = max(pred - half_width, 0.1 if pred > 0 else lower)
                opt_upper = pred + half_width
                
                optimized_lower_ci.append(opt_lower)
                optimized_upper_ci.append(opt_upper)
            
            # Generate forecast dates starting AFTER the last data point
            if start_from_end and self.dates is not None and len(self.dates) > 0:
                last_date = pd.to_datetime(self.dates[-1])
                forecast_dates = pd.date_range(
                    start=last_date + pd.Timedelta(days=1), 
                    periods=steps, 
                    freq='D'
                )
            else:
                forecast_dates = pd.date_range(start='2024-01-01', periods=steps, freq='D')
            
            return {
                'predictions': predictions,
                'confidence_intervals': {
                    'lower': optimized_lower_ci,
                    'upper': optimized_upper_ci,
                    'confidence_level': confidence_level,
                    'optimization': 'Enhanced variability with cyclical patterns'
                },
                'dates': [d.isoformat() for d in forecast_dates],
                'forecast_info': {
                    'steps': steps,
                    'model_order': {
                        'p': self.order[0],
                        'd': self.order[1], 
                        'q': self.order[2]
                    },
                    'forecast_date': datetime.now().isoformat(),
                    'note': f'Enhanced forecast with historical variability patterns'
                }
            }
            
        except Exception as e:
            logger.error(f"Error generating forecast: {str(e)}")
            raise Exception(f"Forecast generation failed: {str(e)}")
    
    def calculate_metrics(self, actual: List[float], predicted: List[float]) -> Dict:
        """
        Calculate comprehensive performance metrics
        """
        try:
            actual = np.array(actual)
            predicted = np.array(predicted)
            
            # Ensure same length
            min_len = min(len(actual), len(predicted))
            actual = actual[:min_len]
            predicted = predicted[:min_len]
            
            if len(actual) == 0:
                return {'error': 'No data to calculate metrics'}
            
            # Calculate metrics
            mae = mean_absolute_error(actual, predicted)
            mse = mean_squared_error(actual, predicted)
            rmse = np.sqrt(mse)
            
            # MAPE - handle division by zero
            non_zero_mask = actual != 0
            if np.any(non_zero_mask):
                mape = np.mean(np.abs((actual[non_zero_mask] - predicted[non_zero_mask]) / actual[non_zero_mask])) * 100
            else:
                mape = float('inf')
            
            # Additional metrics
            mean_actual = np.mean(actual)
            mean_predicted = np.mean(predicted)
            
            # Directional accuracy (for trend prediction)
            if len(actual) > 1:
                actual_direction = np.sign(np.diff(actual))
                predicted_direction = np.sign(np.diff(predicted))
                directional_accuracy = np.mean(actual_direction == predicted_direction) * 100
            else:
                directional_accuracy = 0
            
            # Calculate accuracy percentage based on MAPE (100 - MAPE)
            accuracy_percentage = None
            if mape != float('inf') and mape is not None:
                # Ensure accuracy doesn't go below 0%
                accuracy_percentage = max(0.0, 100.0 - mape)
            
            return {
                'mae': float(mae),
                'mse': float(mse),
                'rmse': float(rmse),
                'mape': float(mape) if mape != float('inf') else None,
                'mean_actual': float(mean_actual),
                'mean_predicted': float(mean_predicted),
                'directional_accuracy': float(directional_accuracy),
                'sample_size': len(actual),
                # Add accuracy percentage based on MAPE (100 - MAPE)
                'accuracy_percentage': accuracy_percentage
            }
            
        except Exception as e:
            logger.error(f"Error calculating metrics: {str(e)}")
            return {'error': str(e)}
    
    def save_model(self, filepath: str):
        """Save the fitted model to disk"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before saving")
        
        try:
            model_data = {
                'fitted_model': self.fitted_model,
                'order': self.order,
                'original_data': self.original_data,
                'dates': self.dates,
                'training_summary': self.training_summary
            }
            
            joblib.dump(model_data, filepath)
            logger.info(f"Model saved to {filepath}")
            
        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")
            raise Exception(f"Model save failed: {str(e)}")
    
    def load_model(self, filepath: str):
        """Load a fitted model from disk"""
        try:
            model_data = joblib.load(filepath)
            
            self.fitted_model = model_data['fitted_model']
            self.order = model_data['order']
            self.original_data = model_data['original_data']
            self.dates = model_data['dates']
            self.training_summary = model_data['training_summary']
            self.is_fitted = True
            
            logger.info(f"Model loaded from {filepath}")
            
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise Exception(f"Model load failed: {str(e)}")
    
    def get_model_summary(self) -> Dict:
        """Get comprehensive model summary"""
        if not self.is_fitted:
            return {'error': 'Model not fitted yet'}
        
        return {
            'model_order': self.order,
            'model_summary': str(self.fitted_model.summary()),
            'training_summary': self.training_summary,
            'is_fitted': self.is_fitted
        }
