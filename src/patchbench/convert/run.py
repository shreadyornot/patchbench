import json
import os
import shutil
import time
from dataclasses import asdict, replace
from datetime import datetime
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from patchbench.convert.config import (
    ConvertConfig,
    RunInfo,
    Variant,
    load_convert_config,
    load_subset_wells,
)
from patchbench.convert.reader import read_site
from patchbench.convert.verify import verify_dataset
from patchbench.convert.writer import Hdf5Writer, ZarrWriter

_WRITERS = {Variant.ZARR_SHARDED: ZarrWriter,
            Variant.ZARR_UNSHARDED: ZarrWriter,
            Variant.HDF5: Hdf5Writer}

# get storage costs
def dataset_stats(path: Path) -> dict:
    files = [path] if path.is_file() else [p for p in path.rglob("*") if p.is_file()]
    return {
        "n_files": len(files),
        "bytes": sum(p.stat().st_size for p in files),               # content size
        "bytes_on_disk": sum(p.stat().st_blocks * 512 for p in files),  # space actually used
    }

# run one conversion variant
def single_run(cfg: ConvertConfig, index: pd.DataFrame, file_format: Variant, wells: list[str] | None, overwrite: bool) -> None:
    if wells is not None:
        path = cfg.paths.output_subset/file_format
        _index = index[index["well"].isin(wells)].reset_index()
        if len(_index) == 0:
            raise ValueError(f"No wells from {cfg.paths.subset_wells} were selected")
    else:
        path = cfg.paths.output_full/file_format
        _index = index

    n_channels = len(cfg.array.channel_names)
    # handle sharding
    chunking = cfg.chunking
    if not cfg.variants[file_format.value].to_shard:
        chunking = replace(chunking, shards=None)

    # record time
    read_s = write_s = 0
    t0 = time.perf_counter()

    # record number of fovs
    n_fovs = 0

    # define writer
    writer = _WRITERS[file_format](path=path, wells=_index["well"].tolist(),
                                  plate=cfg.plate,
                                  array=cfg.array,
                                  chunking=chunking,
                                  codec=cfg.codec,
                                  overwrite=overwrite)

    with writer as w:
        groups =  _index.groupby(["well", "site"])
        print(f"{file_format.upper()} ({'full' if wells is None else 'subset'})")
        for (well, site), rows in tqdm(groups, total=groups.ngroups, desc="Converting"):
            # read and record time
            t = time.perf_counter()
            fov = read_site(well, site, rows, cfg.array.shape, cfg.array.dtype, n_channels)
            read_s += time.perf_counter() - t

            # write and record time
            t = time.perf_counter()
            w.write(well, site, fov)
            write_s += time.perf_counter() - t

            n_fovs+=1

    os.sync()
    total_s = time.perf_counter() - t0

    # verify
    verified = verify_dataset(groups, cfg.array, file_format, writer.dataset_path)
    
    run_info = RunInfo(
        status="complete",
        finished=datetime.now().isoformat(),
        n_fovs=n_fovs, 
        read_time=read_s, write_time=write_s, total_time=total_s, 
        **dataset_stats(writer.dataset_path),
        config=asdict(cfg),
        verified=verified
    )

    # save run stats
    (path / "run_info.json").write_text(json.dumps(asdict(run_info), indent=2, default=str))

    # save config

    print("Conversion Status: COMPLETE")

def move_tiffs(cfg: ConvertConfig, index: pd.DataFrame, subset_wells: list[str]):
    # first save run json for source tiffs
    run_info_full = RunInfo(n_fovs=index.groupby(["well", "site"]).ngroups,
                            **dataset_stats(cfg.paths.data_root / "raw" / cfg.plate.name),
                            )
    (cfg.paths.data_root / "raw" / "run_info.json").write_text(json.dumps(asdict(run_info_full), indent=2, default=str))
    # make subset destination
    dest = cfg.paths.output_subset / "tiff" / cfg.plate.name
    dest.mkdir(parents=True, exist_ok=True)
    # and now move source tiff subset
    subset_index = index[index["well"].isin(subset_wells)].reset_index()
    for src in tqdm(subset_index["local_path"], desc="copy subset TIFFs"):
        shutil.copy(src, dest)
    # write run info for subset
    run_info_subset = RunInfo(n_fovs=subset_index.groupby(["well", "site"]).ngroups,
                            **dataset_stats(dest),
                            )
    (cfg.paths.output_subset / "tiff" / "run_info.json").write_text(json.dumps(asdict(run_info_subset), indent=2, default=str))



def main():
    # read config, index and subset wells
    cfg = load_convert_config()
    index = pd.read_parquet(cfg.paths.index)
    subset_wells = load_subset_wells(cfg.paths.subset_wells)
    for wells in [None, subset_wells]:
        for variant in Variant:
            single_run(cfg, index, variant, wells, overwrite=False)
    # handle source tiffs
    move_tiffs(cfg, index, subset_wells)    

if __name__ == "__main__":
    main()