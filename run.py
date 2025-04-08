import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from dproc import filter_by_moteid, filter_by_time_period, load_dataset, resample_data
from inject.fault import HardoverFault, DriftFault, SpikeFault, ErraticFault, StuckFault

def main():
    DATA_PATH = Path.cwd() / 'data/data.txt'

    dataset = load_dataset(DATA_PATH)
    dataset = filter_by_moteid(dataset, ['7-12'])
    dataset = filter_by_time_period(dataset, start_time='2004-03-01', end_time='2004-03-07')
    dataset = resample_data(dataset, interval='5min')

    expected_freq = '5min'
    motes = dataset['moteid'].unique()
    for mote in motes:
        mote_data = dataset[dataset['moteid'] == mote].copy()
        mote_data.sort_values('datetime', inplace=True)
        full_range = pd.date_range(start=mote_data['datetime'].min(),
                                   end=mote_data['datetime'].max(),
                                   freq=expected_freq)
        aligned_data = mote_data.set_index('datetime').reindex(full_range)
        gap_count = aligned_data['temperature'].isna().sum()
        print(f"Mote {mote}: Expected points = {len(full_range)}, Missing = {gap_count}")
        if gap_count > 0:
            missing_times = aligned_data.index[aligned_data['temperature'].isna()]
            print(f"Missing timestamps: {list(missing_times)}") 

    faults = [
        ("Hardover Fault", HardoverFault(chance=0.20, bias_value=10.0)),
        ("Drift Fault", DriftFault(
            max_duration=pd.Timedelta(minutes=60),
            min_duration=pd.Timedelta(minutes=20),
            chance=0.20,
            initial_bias=0.1
        )),
        ("Spike Fault", SpikeFault(chance=0.20, spike_value=3.0, eta=5)),
        ("Erratic Fault", ErraticFault(
            max_duration=pd.Timedelta(minutes=60),
            min_duration=pd.Timedelta(minutes=20),
            chance=0.20, 
            variance_multiplier=2
        )),
        ("Stuck Fault", StuckFault(
            max_duration=pd.Timedelta(minutes=60),
            min_duration=pd.Timedelta(minutes=20),
            chance=0.20,
            stuck_value=None  
        ))
    ]

    mote_ids = dataset['moteid'].unique()
    
    clean_mote = mote_ids[0]
    faulty_motes = mote_ids[1:]
    print(faulty_motes)
    
    cutoff_date = '2004-03-06'
    result_dfs = []
    
    print(f"Leaving mote {clean_mote} clean.")
    clean = dataset[dataset['moteid'] == clean_mote].copy().reset_index(inplace=False, drop=True)
    clean['type'] = 0
    clean['type'] = clean['type'].astype('int16')
    result_dfs.append(clean)

    train_data = dataset[dataset['datetime'] < cutoff_date]
    test_data = dataset[dataset['datetime'] >= cutoff_date]
    
    for idx, mote in enumerate(faulty_motes):
        fault_name, fault_instance = faults[idx % len(faults)]
        print(f"Applying {fault_name} to mote {mote}...")
        
        mote_train_data = train_data[train_data['moteid'] == mote].copy().reset_index(drop=True)
        mote_test_data = test_data[test_data['moteid'] == mote].copy().reset_index(drop=True)
        
        faulty_train_data = fault_instance.apply(mote_train_data, idx % len(faults) + 1)
        faulty_test_data = fault_instance.apply(mote_test_data, idx % len(faults) + 1)
        
        combined_faulty_data = pd.concat([faulty_train_data, faulty_test_data], ignore_index=True)
        result_dfs.append(combined_faulty_data) 

    result_dataset = pd.concat(result_dfs)

    total_count = len(result_dataset)
    injected_count = (result_dataset['type'] != 0).sum()
    injected_percent = (injected_count / total_count) * 100
    print(f"\nTotal data points: {total_count}")
    print(f"Injected data points: {injected_count} ({injected_percent:.2f}%)\n")
    
    print("Fault injection analysis per mote:")
    for mote in result_dataset['moteid'].unique():
        mote_data = result_dataset[result_dataset['moteid'] == mote]
        total = len(mote_data)
        injected = (mote_data['type'] != 0).sum()
        percent = (injected / total) * 100
        print(f"Mote {mote}: Total = {total}, Injected = {injected} ({percent:.2f}%)")


    selected_mote = mote_ids[5]

    print(f"\nPlot fault sections for mote {selected_mote}:")
    
    df_clean = dataset[dataset['moteid'] == selected_mote].copy()
    df_faulty = result_dataset[result_dataset['moteid'] == selected_mote].copy().sort_values(by='datetime').reset_index(drop=True)
    
    fault_mask = df_faulty['type'] != 0
    
    if not fault_mask.any():
        print(f"No faults found for mote {selected_mote}.")
        return
    
    fault_sections = []
    in_fault = False
    start_idx = None
    
    for idx, is_fault in enumerate(fault_mask):
        if is_fault and not in_fault:
            in_fault = True
            start_idx = idx
        elif not is_fault and in_fault:
            in_fault = False
            fault_sections.append((start_idx, idx - 1))
    
    if in_fault:
        fault_sections.append((start_idx, len(df_faulty) - 1))
    
    print(f"Found {len(fault_sections)} fault sections.")
    
    for i, (start_idx, end_idx) in enumerate(fault_sections):
        context_start = max(start_idx - 12, 0)
        context_end = min(end_idx + 6, len(df_faulty) - 1)
        
        section_faulty = df_faulty.iloc[context_start:context_end + 1].copy()
        
        start_time = section_faulty.iloc[0]['datetime']
        end_time = section_faulty.iloc[-1]['datetime']
        section_clean = df_clean[(df_clean['datetime'] >= start_time) & 
                                (df_clean['datetime'] <= end_time)].copy()
        
        fault_start_time = df_faulty.iloc[start_idx]['datetime']
        fault_end_time = df_faulty.iloc[end_idx]['datetime']
        
        fig, ax = plt.subplots(figsize=(12, 6))

        ax.plot(section_clean['datetime'], section_clean['temperature'], 'b-', alpha=0.7, label="Clean Data")
        ax.plot(section_faulty['datetime'], section_faulty['temperature'], 'r-', label="Faulty Data")

        ax.axvspan(fault_start_time, fault_end_time, color='yellow', alpha=0.3, label="Fault Period")

        ax.set_title(f"Combined Data for Mote {selected_mote} - Fault Section {i+1}/{len(fault_sections)}", fontsize=14)
        ax.set_xlabel("Timestamp")
        ax.set_ylabel("Temperature")
        ax.legend(loc='best')
        ax.grid(True, linestyle='--', alpha=0.7)

        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

    result_dataset.to_csv('./data/inject/6S20%5F.csv', index=False)

if __name__ == '__main__':
    main()
