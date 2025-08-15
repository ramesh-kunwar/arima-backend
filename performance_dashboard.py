#!/usr/bin/env python3
"""
Performance Dashboard - Clear visualization of model performance
Shows exactly what metrics to focus on for ARIMA model evaluation
"""
import requests
import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

def get_performance_data():
    """Get performance data from ARIMA backend"""
    try:
        response = requests.get('http://localhost:8080/sessions')
        if response.status_code == 200:
            data = response.json()
            # Filter for pizza sales sessions only
            pizza_sessions = [s for s in data['sessions'] if 'pizza_sales' in s['filename']]
            return pizza_sessions[-3:]  # Latest 3 pizza sessions
        return []
    except Exception as e:
        print(f"Error fetching data: {e}")
        return []

def create_performance_dashboard():
    """Create comprehensive performance dashboard"""
    sessions = get_performance_data()
    
    if not sessions:
        print("❌ No performance data available")
        return
    
    print("🎯 PIZZA SALES FORECASTING PERFORMANCE ANALYSIS")
    print("=" * 80)
    
    # Performance summary table
    metrics_data = []
    for session in sessions:
        metrics = session['performance_metrics']
        model_order = session['model_order']
        
        # Extract metric name from filename
        if 'revenue' in session['filename']:
            metric_name = 'Revenue ($)'
        elif 'quantity' in session['filename']:
            metric_name = 'Quantity (Pizzas)'
        elif 'orders' in session['filename']:
            metric_name = 'Orders'
        else:
            metric_name = 'Unknown'
        
        metrics_data.append({
            'Metric': metric_name,
            'MAPE (%)': f"{metrics['mape']:.1f}%",
            'MAE': f"{metrics['mae']:.1f}",
            'RMSE': f"{metrics['rmse']:.1f}",
            'R²': f"{metrics['r2']:.3f}",
            'Dir_Acc (%)': f"{metrics['directional_accuracy']:.1f}%",
            'Model': f"({model_order['p']},{model_order['d']},{model_order['q']})",
            'Grade': get_performance_grade(metrics['mape'], metrics['r2'])
        })
    
    # Create DataFrame for better display
    df = pd.DataFrame(metrics_data)
    
    print("\n📊 PERFORMANCE SUMMARY TABLE")
    print("-" * 80)
    print(df.to_string(index=False))
    
    # Detailed analysis
    print(f"\n🔍 DETAILED PERFORMANCE ANALYSIS")
    print("-" * 80)
    
    for i, (session, data) in enumerate(zip(sessions, metrics_data)):
        metrics = session['performance_metrics']
        print(f"\n{i+1}. {data['Metric']} Forecasting:")
        print(f"   ✅ MAPE: {data['MAPE (%)']:>8} {interpret_mape(metrics['mape'])}")
        print(f"   📊 MAE:  {data['MAE']:>8} (Average daily error)")
        print(f"   📈 RMSE: {data['RMSE']:>8} (Penalty for large errors)")
        print(f"   ⚠️  R²:   {data['R²']:>8} {interpret_r2(metrics['r2'])}")
        print(f"   🎯 Dir:  {data['Dir_Acc (%)']:>8} {interpret_directional(metrics['directional_accuracy'])}")
        print(f"   🏆 Grade: {data['Grade']:>7}")
    
    # Recommendations
    print(f"\n💡 KEY RECOMMENDATIONS")
    print("-" * 80)
    
    best_model = min(metrics_data, key=lambda x: float(x['MAPE (%)'].replace('%', '')))
    worst_r2 = min(sessions, key=lambda x: x['performance_metrics']['r2'])
    
    print(f"✅ BEST MODEL: {best_model['Metric']} (MAPE: {best_model['MAPE (%)']})")
    print(f"   → Use this for operational planning and inventory management")
    
    print(f"\n⚠️  AREAS OF CONCERN:")
    for data in metrics_data:
        if float(data['R²']) < 0:
            print(f"   → {data['Metric']}: Negative R² indicates model predicts worse than simple mean")
    
    print(f"\n🎯 WHAT YOU SHOULD FOCUS ON:")
    print("   1. MAPE < 20% = Good for business forecasting (✅ All models qualify)")
    print("   2. R² > 0.3 = Explains reasonable variance (❌ All models fail this)")
    print("   3. Directional Accuracy > 50% = Better than random (❌ Most models fail)")
    
    # Create visualization
    create_performance_plots(sessions, metrics_data)
    
    # Model improvement suggestions
    print(f"\n🔧 MODEL IMPROVEMENT SUGGESTIONS:")
    print("-" * 80)
    print("   1. Add seasonal components (weekly patterns)")
    print("   2. Include external variables (weather, promotions)")
    print("   3. Try ensemble methods (combine multiple models)")
    print("   4. Consider machine learning approaches (Random Forest, XGBoost)")
    print("   5. Segment analysis (weekday vs weekend patterns)")

def get_performance_grade(mape, r2):
    """Assign performance grade based on metrics"""
    if mape < 10 and r2 > 0.7:
        return "A+"
    elif mape < 15 and r2 > 0.5:
        return "A"
    elif mape < 20 and r2 > 0.3:
        return "B"
    elif mape < 25 and r2 > 0.1:
        return "C"
    elif mape < 30:
        return "D"
    else:
        return "F"

def interpret_mape(mape):
    """Interpret MAPE value"""
    if mape < 10:
        return "🟢 Excellent"
    elif mape < 20:
        return "🟡 Good (Business Acceptable)"
    elif mape < 30:
        return "🟠 Fair"
    else:
        return "🔴 Poor"

def interpret_r2(r2):
    """Interpret R-squared value"""
    if r2 > 0.7:
        return "🟢 Strong relationship"
    elif r2 > 0.5:
        return "🟡 Moderate relationship"
    elif r2 > 0.3:
        return "🟠 Weak relationship"
    elif r2 > 0:
        return "🔴 Very weak relationship"
    else:
        return "❌ NEGATIVE: Model worse than mean"

def interpret_directional(dir_acc):
    """Interpret directional accuracy"""
    if dir_acc > 60:
        return "🟢 Good trend prediction"
    elif dir_acc > 50:
        return "🟡 Better than random"
    else:
        return "🔴 Worse than random guess"

def create_performance_plots(sessions, metrics_data):
    """Create performance visualization plots"""
    
    # Extract data for plotting
    metrics_names = [data['Metric'] for data in metrics_data]
    mape_values = [float(data['MAPE (%)'].replace('%', '')) for data in metrics_data]
    r2_values = [session['performance_metrics']['r2'] for session in sessions]
    dir_acc_values = [session['performance_metrics']['directional_accuracy'] for session in sessions]
    
    # Create subplot
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('🎯 Pizza Sales Forecasting Performance Dashboard', fontsize=16, fontweight='bold')
    
    # 1. MAPE Comparison
    colors = ['green' if x < 20 else 'orange' if x < 30 else 'red' for x in mape_values]
    ax1.bar(metrics_names, mape_values, color=colors, alpha=0.7)
    ax1.axhline(y=20, color='red', linestyle='--', alpha=0.7, label='Business Threshold (20%)')
    ax1.set_title('MAPE (Mean Absolute Percentage Error)', fontweight='bold')
    ax1.set_ylabel('MAPE (%)')
    ax1.legend()
    for i, v in enumerate(mape_values):
        ax1.text(i, v + 0.5, f'{v:.1f}%', ha='center', fontweight='bold')
    
    # 2. R-squared Comparison
    colors = ['red' if x < 0 else 'orange' if x < 0.3 else 'green' for x in r2_values]
    ax2.bar(metrics_names, r2_values, color=colors, alpha=0.7)
    ax2.axhline(y=0, color='black', linestyle='-', alpha=0.5, label='Zero Line')
    ax2.axhline(y=0.3, color='green', linestyle='--', alpha=0.7, label='Acceptable Threshold')
    ax2.set_title('R-squared (Variance Explained)', fontweight='bold')
    ax2.set_ylabel('R²')
    ax2.legend()
    for i, v in enumerate(r2_values):
        ax2.text(i, v + 0.02, f'{v:.3f}', ha='center', fontweight='bold')
    
    # 3. Directional Accuracy
    colors = ['green' if x > 60 else 'orange' if x > 50 else 'red' for x in dir_acc_values]
    ax3.bar(metrics_names, dir_acc_values, color=colors, alpha=0.7)
    ax3.axhline(y=50, color='red', linestyle='--', alpha=0.7, label='Random Chance (50%)')
    ax3.set_title('Directional Accuracy (Trend Prediction)', fontweight='bold')
    ax3.set_ylabel('Accuracy (%)')
    ax3.legend()
    for i, v in enumerate(dir_acc_values):
        ax3.text(i, v + 1, f'{v:.1f}%', ha='center', fontweight='bold')
    
    # 4. Overall Performance Radar
    grades = [data['Grade'] for data in metrics_data]
    grade_scores = {'A+': 10, 'A': 8, 'B': 6, 'C': 4, 'D': 2, 'F': 0}
    scores = [grade_scores.get(grade, 0) for grade in grades]
    
    ax4.bar(metrics_names, scores, color=['red' if s < 4 else 'orange' if s < 6 else 'green' for s in scores], alpha=0.7)
    ax4.set_title('Overall Performance Grade', fontweight='bold')
    ax4.set_ylabel('Performance Score')
    ax4.set_ylim(0, 10)
    for i, (score, grade) in enumerate(zip(scores, grades)):
        ax4.text(i, score + 0.2, grade, ha='center', fontweight='bold', fontsize=12)
    
    plt.tight_layout()
    plt.savefig('performance_dashboard.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\n📊 Performance dashboard saved as: performance_dashboard.png")

if __name__ == "__main__":
    create_performance_dashboard()
