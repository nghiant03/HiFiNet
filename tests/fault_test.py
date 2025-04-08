import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

from inject.fault import HardoverFault, DriftFault, SpikeFault, ErraticFault, StuckFault

def generate_test_data(n_points=1000):
    """Generate sample sensor data for testing"""
    start_time = datetime(2023, 1, 1)
    times = [start_time + timedelta(minutes=i) for i in range(n_points)]
    
    t = np.linspace(0, 10 * np.pi, n_points)
    normal_signal = np.sin(t) * 10 + 20 
    noise = np.random.normal(0, 0.5, n_points)
    normal_signal = normal_signal + noise
    
    df = pd.DataFrame({
        'datetime': times,
        'sensor_value': normal_signal
    })
    
    return df

def visualize_faults(original_df, faulty_dfs, fault_names):
    """Plot original and faulty signals for comparison"""
    n_faults = len(faulty_dfs)
    fig, axes = plt.subplots(n_faults + 1, 1, figsize=(12, 3 * (n_faults + 1)), sharex=True)
    
    axes[0].plot(original_df['datetime'], original_df['sensor_value'], 'b-')
    axes[0].set_title('Original Signal')
    axes[0].set_ylabel('Value')
    axes[0].grid(True)
    
    for i, (faulty_df, name) in enumerate(zip(faulty_dfs, fault_names)):
        axes[i+1].plot(original_df['datetime'], original_df['sensor_value'], 'b-', alpha=0.3)
        axes[i+1].plot(faulty_df['datetime'], faulty_df['sensor_value'], 'r-')
        axes[i+1].set_title(f'{name} Signal')
        axes[i+1].set_ylabel('Value')
        axes[i+1].grid(True)
    
    axes[-1].set_xlabel('Time')
    plt.tight_layout()
    return fig

def main():
    print("Generating test data...")
    df = generate_test_data(n_points=1000)
    
    faults = [
        ("Hardover Fault", HardoverFault(chance=0.05, bias_value=15.0)),
        ("Drift Fault", DriftFault(
            max_duration=pd.Timedelta(minutes=100),
            min_duration=pd.Timedelta(minutes=50),
            chance=0.1,
            initial_bias=0.2
        )),
        ("Spike Fault", SpikeFault(chance=0.05, spike_value=20.0, eta=5)),
        ("Erratic Fault", ErraticFault(chance=0.1, variance_multiplier=10.0)),
        ("Stuck Fault", StuckFault(
            max_duration=pd.Timedelta(minutes=80),
            min_duration=pd.Timedelta(minutes=30),
            chance=0.08,
            stuck_value=None  
        ))
    ]
    
    faulty_dfs = []
    fault_names = []
    
    for name, fault in faults:
        print(f"Applying {name}...")
        faulty_df = fault.apply(df)
        faulty_dfs.append(faulty_df)
        fault_names.append(name)
    
    fig = visualize_faults(df, faulty_dfs, fault_names)
    plt.savefig('sensor_faults_visualization.png')
    plt.close(fig)
    print("Visualization saved as 'sensor_faults_visualization.png'")
    
    print("\nFault Statistics:")
    for name, original, faulty in zip(fault_names, [df] * len(faults), faulty_dfs):
        diff = (faulty['sensor_value'] != original['sensor_value']).sum()
        percent = diff / len(original) * 100
        print(f"{name}: {diff} points affected ({percent:.2f}% of data)")

if __name__ == "__main__":
    main()
