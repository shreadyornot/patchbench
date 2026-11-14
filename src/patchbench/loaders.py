import h5py
import numpy as np
import zarr

from patchbench.convert.config import Variant

FULL = (slice(None), )

def _get_field_path(well: str, site: int) -> np.ndarray:
    return f"{well[0]}/{well[1:]}/{site-1}"

"Read regions of a field of view back from zarr or h5py open file"
def read_region(handle: zarr.Group | h5py.File, well: str, site: str, file_format: Variant, region: tuple = FULL) -> np.ndarray:
    if file_format.is_zarr:
        arr = handle[f"{_get_field_path(well, site)}/0"]
    elif file_format == Variant.HDF5:
        arr = handle[_get_field_path(well, site)]
    else:
        raise ValueError("Format not handled")
    return arr[region]