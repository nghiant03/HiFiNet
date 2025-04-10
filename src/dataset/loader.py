import logging
import pandas as pd
from pathlib import Path

from src.dataset.args import IntelArgs
from src.dataset.filter import filter_by_moteid, filter_by_time
logger = logging.getLogger(__name__)

def load_intel_dataset(file_path: Path, args: IntelArgs) -> pd.DataFrame:
    logger.info("Loading Intel dataset.")
    columns = ['date', 'time', 'epoch', 'moteid', 'temperature', 'humidity', 'light', 'voltage']

    try:
        data = pd.read_csv(file_path, sep=' ', names=columns, header=None)
    except FileNotFoundError:
        logger.error("Intel dataset not found at {file_path.as_posix()}!")
        raise 

    data['datetime'] = pd.to_datetime(data['date'] + ' ' + data['time'], format='mixed')
    data = filter_by_time(data, start_time=args.start_date, end_time=args.end_date)

    nan_count = data.isna().sum().sum()
    if nan_count > 0:
        logger.debug(f"Initial NaN count in raw data: {nan_count}")
        logger.debug(f"NaN counts per column:\n{data.isna().sum()}")
    else:
        logger.debug("No NaN values found in the initial raw data.")

    data = data.dropna()
    data = data.drop_duplicates()

    data = data.drop(['date', 'time', 'humidity', 'light', 'voltage', 'epoch'], axis=1)

    data['moteid'] = data['moteid'].astype('int16')
    data = filter_by_moteid(data, motes=args.mote_id)
    data = resample_data(data, interval=args.interval)

    return data

def resample_data(data, interval: str) -> pd.DataFrame:
    logger.info("Starting resample for datajj")
    data['datetime'] = pd.to_datetime(data['datetime'])
    resampled_list = []
    
    unique_motes = data['moteid'].unique()
    
    logger.debug(f"Found {len(unique_motes)} unique mote IDs for resampling.")
    for mote in unique_motes:
        mote_data = data[data['moteid'] == mote].copy()
        mote_data = mote_data.set_index('datetime')
        
        resampled_mote = mote_data.resample(interval).mean()
        resampled_mote['moteid'] = mote
        
        nan_before = resampled_mote['temperature'].isna().sum()
        if nan_before > 0:
            logger.debug(f"Mote {mote}: Found {nan_before} NaN values in 'temperature' after resampling, before interpolation.")

        resampled_mote['temperature'] = resampled_mote['temperature'].interpolate('time')


        nan_after = resampled_mote['temperature'].isna().sum()
        
        filled_count = nan_before - nan_after
        if filled_count > 0:
            logger.debug(f"Mote {mote}: Interpolated {filled_count} missing 'temperature' values.")
        elif nan_before > 0 and nan_after > 0:
            logger.warning(f"Mote {mote}: {nan_after} NaNs remain in 'temperature' after interpolation (likely boundary NaNs).")

        resampled_mote = resampled_mote.reset_index()
        resampled_list.append(resampled_mote)
    
    resampled_data = pd.concat(resampled_list, ignore_index=True)

    return resampled_data
