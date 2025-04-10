import pandas as pd
from typing import List, Optional, Union

def filter_by_moteid(data, motes: List[str]) -> pd.DataFrame:
    filtered_data = []

    for mote in motes:
        if '-' in mote:
            first, last = mote.split('-')
            mote_data = data[(data['moteid'] >= int(first)) & (data['moteid'] <= int(last))]
        else:
            mote_data = data[data['moteid'] == int(mote)]
        filtered_data.append(mote_data)

    filtered_data = pd.concat(filtered_data)

    return filtered_data.reset_index(inplace=False, drop=True)

def filter_by_time(data, start_time: Optional[Union[pd.Timestamp, str]] = None, end_time: Optional[Union[pd.Timestamp, str]] = None) -> pd.DataFrame:
    if isinstance(start_time, str):
        start_time = pd.to_datetime(start_time)
    if isinstance(end_time, str):
        end_time = pd.to_datetime(end_time)

    filtered_data = data.copy()

    if start_time:
        filtered_data = filtered_data[filtered_data['datetime'] >= start_time]

    if end_time:
        filtered_data = filtered_data[filtered_data['datetime'] <= end_time]

    filtered_data = filtered_data.sort_values(by='datetime')
    return filtered_data.reset_index(inplace=False, drop=True)
