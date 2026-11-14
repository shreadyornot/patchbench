from pathlib import Path

import h5py
import hdf5plugin  # noqa: F401 (registers Blosc so h5py can decompress)
import numpy as np
import zarr
from pandas.api.typing import DataFrameGroupBy
from tqdm import tqdm

from patchbench.convert.config import ArraySpec, Variant
from patchbench.convert.reader import read_site
from patchbench.loaders import read_region


def verify_dataset(groups: DataFrameGroupBy, array_spec: ArraySpec, file_format: Variant, dataset_path: Path) -> dict:
    "Compare every converted data with the source from Tiffs"
    if file_format.is_zarr:
        handle = zarr.open_group(dataset_path, mode="r")
    elif file_format == Variant.HDF5:
        handle = h5py.File(dataset_path, "r")
    else:
        raise ValueError("Format not handled")

    mismatches, missing = [], []
    try:
        for (well, site), rows in tqdm(groups, total=groups.ngroups, desc="Verifying"):
            expected = read_site(well, site, rows, array_spec.shape, array_spec.dtype, len(array_spec.channel_names))
            try:
                actual = read_region(handle, well, site, file_format)
            except KeyError:
                missing.append((well,site))
                continue
            if actual.dtype != expected.dtype or not np.array_equal(actual, expected):
                mismatches.append((well, site))
    finally:
        if file_format == Variant.HDF5:
            handle.close()

    return {
        "ok": not missing and not mismatches,
        "checked": groups.ngroups,
        "missing": missing,
        "mismatches": mismatches
    }