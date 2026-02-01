import csv
import pandas as pd
import numpy as np
from collections import defaultdict
import os

# File paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_TRANSFORMER = os.path.join(BASE_DIR, "inference_log_transformer.csv")
LOG_BILSTM = os.path.join(BASE_DIR, "inference_log_bilstm.csv")
LOG_BILSTM_TRANSFORMER = os.path.join(BASE_DIR, "inference_log_bilstm_transformer.csv")

CLASSES = ['down', 'left', 'right', 'up']

def load_log(filepath):
    """Load inference log CSV"""
    try:
        df = pd.read_csv(filepath)
        return df
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return None

def calculate_metrics(df, model_name):
    """Calculate metrics for a model"""
    if df is None or len(df) == 0:
        print(f"No data for {model_name}")
        return None
    
    metrics = {
        'model': model_name,
        'total_samples': len(df),
        'per_class': {},
        'overall': {}
    }
    
    # Per-class metrics
    for cls in CLASSES:
        cls_data = df[df['Prediksi Model'] == cls]
        if len(cls_data) > 0:
            inf_time = cls_data['Inference Time (ms)'].astype(float).mean()
            transport_time = cls_data['Transport Latency (ms)'].astype(float).mean()
            metrics['per_class'][cls] = {
                'count': len(cls_data),
                'avg_confidence': cls_data['Confidence'].astype(float).mean(),
                'min_confidence': cls_data['Confidence'].astype(float).min(),
                'max_confidence': cls_data['Confidence'].astype(float).max(),
                'std_confidence': cls_data['Confidence'].astype(float).std(),
                'avg_inference_time': inf_time,
                'avg_transport_latency': transport_time,
                'avg_total_response': inf_time + transport_time,
            }
    
    # Overall metrics
    if len(df) > 0:
        inf_time = df['Inference Time (ms)'].astype(float).mean()
        transport_time = df['Transport Latency (ms)'].astype(float).mean()
        metrics['overall'] = {
            'avg_confidence': df['Confidence'].astype(float).mean(),
            'min_confidence': df['Confidence'].astype(float).min(),
            'max_confidence': df['Confidence'].astype(float).max(),
            'std_confidence': df['Confidence'].astype(float).std(),
            'avg_inference_time': inf_time,
            'avg_transport_latency': transport_time,
            'avg_total_response': inf_time + transport_time,
            'success_rate': (df['Game Status'] == 'success').sum() / len(df) * 100,
        }
    
    return metrics

def print_comparison(all_metrics):
    """Print latency and inference performance comparison"""
    print("\n" + "="*120)
    print("PERBANDINGAN LATENCY & INFERENCE PERFORMANCE - 40 MFCC COEFFICIENTS")
    print("="*120)
    
    # Inference speed comparison
    print("\ INFERENCE TIME (ms) - Model Execution Speed")
    print("-" * 120)
    print(f"{'Model':<30} {'Avg Inf Time':<15} {'Min':<10} {'Max':<10} {'Std Dev':<10} {'Per-Sample':<12}")
    print("-" * 120)
    
    for metrics in all_metrics:
        model = metrics['model']
        inf_times = []
        for cls in CLASSES:
            if cls in metrics['per_class']:
                inf_times.append(metrics['per_class'][cls]['avg_inference_time'])
        
        avg_inf = metrics['overall']['avg_inference_time']
        min_inf = min(inf_times) if inf_times else 0
        max_inf = max(inf_times) if inf_times else 0
        
        # Calculate std dev for inference times across samples
        all_inf = []
        for cls in CLASSES:
            if cls in metrics['per_class']:
                all_inf.append(metrics['per_class'][cls]['avg_inference_time'])
        std_inf = np.std(all_inf) if all_inf else 0
        
        print(f"{model:<30} {avg_inf:<15.1f} {min_inf:<10.1f} {max_inf:<10.1f} {std_inf:<10.2f} {avg_inf/15:<12.2f}")
    
    # End-to-end latency
    print("\n\nTOTAL RESPONSE TIME (ms) - End-to-End Latency")
    print("-" * 120)
    print(f"{'Model':<30} {'Avg Total':<18} {'Min':<12} {'Max':<12} {'Inference %':<15} {'Transport %':<15}")
    print("-" * 120)
    
    for metrics in all_metrics:
        model = metrics['model']
        total_resp = metrics['overall']['avg_total_response']
        
        resp_times = []
        for cls in CLASSES:
            if cls in metrics['per_class']:
                resp_times.append(metrics['per_class'][cls]['avg_total_response'])
        
        min_resp = min(resp_times) if resp_times else 0
        max_resp = max(resp_times) if resp_times else 0
        
        avg_inf = metrics['overall']['avg_inference_time']
        inf_percent = (avg_inf / total_resp * 100) if total_resp > 0 else 0
        transport_percent = 100 - inf_percent
        
        print(f"{model:<30} {total_resp:<18.2f} {min_resp:<12.2f} {max_resp:<12.2f} {inf_percent:<15.1f}% {transport_percent:<15.1f}%")
    
    # Per-class inference latency
    print("\n\nPER-CLASS INFERENCE TIME (ms)")
    print("="*120)
    
    for cls in CLASSES:
        print(f"\nKelas: {cls.upper()}")
        print("-" * 120)
        print(f"{'Model':<30} {'Avg (ms)':<15} {'Min (ms)':<15} {'Max (ms)':<15} {'Std Dev':<15} {'Count':<10}")
        print("-" * 120)
        
        for metrics in all_metrics:
            if cls in metrics['per_class']:
                cls_info = metrics['per_class'][cls]
                model = metrics['model']
                avg_inf = cls_info['avg_inference_time']
                min_inf = cls_info['min_confidence']
                max_inf = cls_info['max_confidence']
                std_inf = cls_info['std_confidence']
                count = cls_info['count']
                
                print(f"{model:<30} {avg_inf:<15.3f} {min_inf:<15.3f} {max_inf:<15.3f} {std_inf:<15.4f} {count:<10}")
    
    # Ranking by speed
    print("\n\nRANKING BY INFERENCE SPEED")
    print("="*120)
    
    sorted_by_speed = sorted(all_metrics, key=lambda x: x['overall']['avg_inference_time'])
    
    for i, metrics in enumerate(sorted_by_speed, 1):
        model = metrics['model']
        avg_inf = metrics['overall']['avg_inference_time']
        baseline = sorted_by_speed[0]['overall']['avg_inference_time']
        slowdown = (avg_inf / baseline - 1) * 100 if i > 1 else 0
        
        if i == 1:
            print(f"{i}. ⚡ {model:<30} {avg_inf:.3f} ms (BASELINE)")
        else:
            print(f"{i}. {model:<32} {avg_inf:.3f} ms (+{slowdown:.1f}% slower)")
    
    print("\n" + "="*120 + "\n")

def save_comparison_csv(all_metrics):
    """Save comparison results to separate CSV files"""
    output_overall = os.path.join(BASE_DIR, "model_comparison_40coeff_overall.csv")
    output_perclass = os.path.join(BASE_DIR, "model_comparison_40coeff_perclass.csv")
    
    # Overall metrics
    rows_overall = []
    for metrics in all_metrics:
        rows_overall.append({
            'Model': metrics['model'],
            'Total_Samples': metrics['total_samples'],
            'Avg_Confidence_%': f"{metrics['overall']['avg_confidence']*100:.2f}%",
            'Std_Confidence_%': f"{metrics['overall']['std_confidence']*100:.2f}%",
            'Min_Confidence_%': f"{metrics['overall']['min_confidence']*100:.2f}%",
            'Max_Confidence_%': f"{metrics['overall']['max_confidence']*100:.2f}%",
            'Avg_Inference_Time_ms': f"{metrics['overall']['avg_inference_time']:.1f}",
            'Avg_Transport_Latency_ms': f"{metrics['overall']['avg_transport_latency']:.2f}",
            'Avg_Total_Response_ms': f"{metrics['overall']['avg_total_response']:.2f}",
            'Success_Rate': f"{metrics['overall']['success_rate']:.2f}%",
        })
    
    df_overall = pd.DataFrame(rows_overall)
    df_overall.to_csv(output_overall, index=False)
    print(f"Overall metrics: {output_overall}")
    
    # Per-class metrics
    rows_perclass = []
    for metrics in all_metrics:
        for cls in CLASSES:
            if cls in metrics['per_class']:
                cls_info = metrics['per_class'][cls]
                rows_perclass.append({
                    'Model': metrics['model'],
                    'Class': cls,
                    'Count': cls_info['count'],
                    'Avg_Confidence_%': f"{cls_info['avg_confidence']*100:.2f}%",
                    'Std_Confidence_%': f"{cls_info['std_confidence']*100:.2f}%",
                    'Min_Confidence_%': f"{cls_info['min_confidence']*100:.2f}%",
                    'Max_Confidence_%': f"{cls_info['max_confidence']*100:.2f}%",
                    'Avg_Inference_Time_ms': f"{cls_info['avg_inference_time']:.1f}",
                    'Avg_Transport_Latency_ms': f"{cls_info['avg_transport_latency']:.2f}",
                    'Avg_Total_Response_ms': f"{cls_info['avg_total_response']:.2f}",
                })
    
    df_perclass = pd.DataFrame(rows_perclass)
    df_perclass.to_csv(output_perclass, index=False)
    print(f"Per-class metrics: {output_perclass}\n")

# Main execution
if __name__ == "__main__":
    print("\nLoading inference logs...")
    
    df_transformer = load_log(LOG_TRANSFORMER)
    df_bilstm = load_log(LOG_BILSTM)
    df_bilstm_transformer = load_log(LOG_BILSTM_TRANSFORMER)
    
    print("Logs loaded\n")
    
    print("Menghitung metrics...")
    
    metrics_transformer = calculate_metrics(df_transformer, "MFCC-Transformer")
    metrics_bilstm = calculate_metrics(df_bilstm, "MFCC-LSTM (BiLSTM)")
    metrics_bilstm_transformer = calculate_metrics(df_bilstm_transformer, "BiLSTM-Transformer")
    
    all_metrics = [metrics_transformer, metrics_bilstm, metrics_bilstm_transformer]
    
    print("Metrics calculated\n")
    
    # Print comparison
    print_comparison(all_metrics)
    
    # Save to CSV
    save_comparison_csv(all_metrics)
    
    print("Perbandingan selesai!")
