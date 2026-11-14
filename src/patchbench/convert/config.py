from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

import yaml


class Variant(StrEnum):
    ZARR_SHARDED = "zarr_sharded"
    ZARR_UNSHARDED = "zarr_unsharded"
    HDF5 = "hdf5"

    @property
    def is_zarr(self) -> bool:
        return self in (Variant.ZARR_SHARDED, Variant.ZARR_UNSHARDED)

def project_root() -> Path:
    "Walk up from this to find pyproject.toml"
    here = Path(__file__).resolve()
    for folder in [here, *here.parents]:
        if (folder/"pyproject.toml").exists():
            return folder
    raise FileNotFoundError("Could not find project root (no pyproject.toml)")

@dataclass(frozen=True)
class Paths:
    data_root: Path
    index: Path
    output_full: Path
    output_subset: Path
    subset_wells: Path

@dataclass(frozen=True)
class Plate:
    name: str
    nrows: int
    ncols: int
    field_count: int

@dataclass(frozen=True)
class ArraySpec:
    axes: tuple[int, ...]
    dtype: str
    shape: tuple[int, int]
    fill_value: int
    channel_names: list[str]

    def __post_init__(self):
        object.__setattr__(self, "shape", tuple(self.shape))

@dataclass
class Chunking:
    chunks: tuple[int, ...]
    shards: tuple[int, ...] | None

    def __post_init__(self):
        if len(self.chunks) != 5 or (self.shards is not None and len(self.shards) != 5):
            raise ValueError("chunks and shards need 5 values (t, c, z, y, x)")

@dataclass(frozen=True)
class Codec:
    name: str
    cname: str
    clevel: int
    shuffle: str
    nthreads: int

@dataclass(frozen=True)
class VariantSpec:
    file_format: str
    to_shard: bool = False

@dataclass(frozen=True)
class ConvertConfig:
    paths: Paths
    plate: Plate
    array: ArraySpec
    chunking: Chunking
    codec: Codec
    ome_ngff_version: str
    field_path: str
    hdf5_layout: str
    variants: dict[str, VariantSpec]

# -------------- the loader -------------------

def load_convert_config(path: str | Path = "configs/phase1_convert.yaml") -> ConvertConfig:
    root = project_root()
    with open(root / path) as f:
        raw = yaml.safe_load(f)

    return ConvertConfig(
        paths=Paths(**{key: root/value for key, value in raw["paths"].items()}),
        plate=Plate(**raw["plate"]),
        array=ArraySpec(**raw["array"]),
        chunking=Chunking(**raw["chunking"]),
        codec=Codec(**raw["codec"]),
        ome_ngff_version=raw["ome_ngff"]["version"],
        field_path=raw["ome_ngff"]["field_path"],
        hdf5_layout=raw["hdf5"]["layout"],
        variants={name: VariantSpec(**v) for name, v in raw["variants"].items()},
    )

def load_subset_wells(path: Path) -> list:
    with open(path) as f:
        return yaml.safe_load(f)["wells"]

@dataclass(frozen=True, kw_only=True)
class RunInfo:
    status: str | None = None
    finished: str | None = None
    n_fovs: int
    read_time: float | None = None
    write_time: float | None = None
    total_time: float | None = None
    n_files: int
    bytes: int
    bytes_on_disk: int
    config: dict | None = None
    verified: bool | None = None