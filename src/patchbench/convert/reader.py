import numpy as np
import pandas as pd
import tifffile


def read_site(well: str, site: str, site_group: pd.DataFrame,
             array_shape: tuple[int, int], array_dtype: str, n_channels: int) -> np.ndarray:
    if len(site_group) != 5:
        raise ValueError(f"Expected 5 channels for Well: {well}|Site: {site}, but got {len(site_group)}")
    # sort again for safety
    site_group = site_group.sort_values(by='channel')
    site_across_channels = []
    for _, row in site_group.iterrows():
        # read 
        channel_array = tifffile.imread(row.local_path)
        # check shape and data type
        if channel_array.shape != array_shape:
            raise ValueError(f"Expected image at Well: {well}|Site: {site}|Channel: {row.channel}({row.dye}) to be 1080X1080 but got {channel_array.shape}")
        if channel_array.dtype != array_dtype:
            raise ValueError(f"Expected image at Well: {well}|Site: {site}|Channel: {row.channel}({row.dye}) to be uint16 but got {channel_array.dtype}")
        # add to list
        site_across_channels.append(channel_array)
    # stack channels
    stacked = np.stack(site_across_channels, axis=0)
    # add time and z to make (T, C, Z, Y, X)
    final = np.expand_dims(stacked, axis=(0, 2))
    assert final.shape == (1, n_channels, 1) + array_shape
    return final