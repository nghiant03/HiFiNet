import pandas as pd
from pathlib import Path

def load_dataset(file_path: Path):
    columns = ['date', 'time', 'epoch', 'moteid', 'temperature', 'humidity', 'light', 'voltage']

    data = pd.read_csv(file_path, sep=' ', names=columns, header=None)

    data = data.dropna()
    data = data.drop_duplicates()

    data['datetime'] = pd.to_datetime(data['date'] + ' ' + data['time'], format='mixed')
    data = data.drop(['date', 'time', 'humidity', 'light', 'voltage', 'epoch'], axis=1)

    data['moteid'] = data['moteid'].astype('int16')

    return data

def resample_data(data, interval='5min'):
    data['datetime'] = pd.to_datetime(data['datetime'])
    resampled_list = []
    
    unique_motes = data['moteid'].unique()
    
    for mote in unique_motes:
        mote_data = data[data['moteid'] == mote].copy()
        mote_data = mote_data.set_index('datetime')
        
        resampled_mote = mote_data.resample(interval).mean()
        resampled_mote['moteid'] = mote
        
        resampled_mote['temperature'] = resampled_mote['temperature'].interpolate('time')
        resampled_mote = resampled_mote.reset_index()
        resampled_list.append(resampled_mote)
    
    resampled_data = pd.concat(resampled_list, ignore_index=True)

    return resampled_data
