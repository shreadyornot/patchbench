import json
import os
from abc import ABC, abstractmethod
from pathlib import Path

import h5py
import hdf5plugin
import numcodecs
import numpy as np
import zarr
from zarr.codecs import BloscCodec

from patchbench.convert.config import (
    ArraySpec,
    Chunking,
    Codec,
    Plate,
)


# Function to build PLATE(only used for HCS) metadata
def _make_plate_metadata(name: str, 
                        wells: list[str],
                        nrows: int,
                        ncols: int,
                        field_count: int, 
                        ) -> dict:
    rows = [chr(ord("A") + i) for i in range(nrows)]
    cols = [f"{i+1:02d}" for i in range(ncols)] 
    return {
        "version": "0.5",
        "plate": {
            "name": name,
            "rows": [{"name": r} for r in rows],
            "columns": [{"name": c} for c in cols],
            "wells": [
                {"path": f"{w[0]}/{w[1:]}",
                 "rowIndex": rows.index(w[0]),
                 "columnIndex": cols.index(w[1:])}
                for w in sorted(wells)
            ],
            "field_count": field_count,
        },
    }

# Function to build well metadata
def _make_well_metadata(field_count: int) -> dict:
    return {"version": "0.5",
            "well": {"images": [{"path": str(i)} for i in range(field_count)]}}

# function to build image metadata
def _make_image_metadata(channel_names: list[str]) -> dict:
    return {
        "version": "0.5",
        "multiscales": [{
            "axes": [
                {"name": "t", "type": "time"},
                {"name": "c", "type": "channel"},
                {"name": "z", "type": "space"},
                {"name": "y", "type": "space"},
                {"name": "x", "type": "space"},
            ],
            "datasets": [{"path": "0",
                          "coordinateTransformations": [{"type": "scale", "scale": [1, 1, 1, 1, 1]}]}],
        }],
        "omero": {"channels": [{"label": n} for n in channel_names]},
    }

class BaseWriter(ABC):
    def __init__(self,
                path: Path, wells: list[str],
                plate: Plate,
                array: ArraySpec,
                chunking: Chunking,
                codec: Codec,
                overwrite=False):
        self.path = path
        self.wells = wells
        self.plate = plate
        self.array = array
        self.chunking = chunking
        self.overwrite = overwrite
        self.compressor = self._make_compressor(codec)
        self.nthreads = codec.nthreads

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.finish()

    @abstractmethod
    def _make_compressor(self, codec):
        "define compressor settings"

    @abstractmethod
    def start(self):
        "Setup"

    @abstractmethod
    def write(self, well, site, site_array):
        "Write"

    @abstractmethod
    def finish(self):
        "close"


class ZarrWriter(BaseWriter):

    def _make_compressor(self, codec: Codec):
        return BloscCodec(cname=codec.cname, clevel=codec.clevel, shuffle=codec.shuffle)

    def start(self):
        self.dataset_path: Path = self.path/f"{self.plate.name}.ome.zarr"
        numcodecs.blosc.set_nthreads(self.nthreads) # parallelism comes from the DataLoader, not Blosc
        self.root = zarr.create_group(
            store=self.dataset_path, zarr_format=3, overwrite=self.overwrite,
            attributes={"ome": _make_plate_metadata(name=self.plate.name, 
                                                   wells=self.wells, 
                                                   nrows=self.plate.nrows,
                                                   ncols=self.plate.ncols,
                                                   field_count=self.plate.field_count)}
        )

    def write(self, well: str, site: int, site_array: np.ndarray):
        row, col = well[0], well[1:]
        well_grp = self.root.require_group(row).require_group(col)
        if "ome" not in well_grp.attrs:
            well_grp.attrs["ome"] = _make_well_metadata(self.plate.field_count)
        fov_grp = well_grp.create_group(
            str(site - 1), attributes={"ome": _make_image_metadata(self.array.channel_names)}
        )
        arr = fov_grp.create_array("0", 
                                   shape=site_array.shape, 
                                   dtype=site_array.dtype,
                                   chunks=self.chunking.chunks, shards=self.chunking.shards,
                                   compressors=self.compressor, 
                                   fill_value=self.array.fill_value,
                                   dimension_names=self.array.axes,
                                   )
        arr[:] = site_array

    def finish(self):
        pass


_SHUFFLE = {
    "noshuffle": hdf5plugin.Blosc.NOSHUFFLE,
    "shuffle": hdf5plugin.Blosc.SHUFFLE,
    "bitshuffle": hdf5plugin.Blosc.BITSHUFFLE,
}

class Hdf5Writer(BaseWriter):

    def _make_compressor(self, codec: Codec):
        return hdf5plugin.Blosc(
            cname=codec.cname, clevel=codec.clevel, shuffle=_SHUFFLE[codec.shuffle]
        )

    def start(self):
        os.environ["BLOSC_NTHREADS"] = str(self.nthreads)
        self.dataset_path: Path = self.path/f"{self.plate.name}.h5"
        # dumpy plate metadata to a single file
        self.path.mkdir(parents=True, exist_ok=True)
        self.file = h5py.File(self.dataset_path, "w" if self.overwrite else "w-")

        self.file.attrs["ome"] = json.dumps(
            _make_plate_metadata(self.plate.name, self.wells, self.plate.nrows,
                                 self.plate.ncols, self.plate.field_count)
        )
        self.file.attrs["channel_names"] = self.array.channel_names
        self.file.attrs["axes"] = self.array.axes

    def write(self, well, site, site_array):
        row, col = well[0], well[1:]
        well_grp = self.file.require_group(f"{row}/{col}")
        dset = well_grp.create_dataset(
            str(site - 1),
            data=site_array,
            chunks=tuple(self.chunking.chunks),
            fillvalue=self.array.fill_value,
            **self.compressor
        )
        for i, name in enumerate(self.array.axes):
            dset.dims[i].label = name

    def finish(self):
        self.file.close()