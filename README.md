# Benchmarking Bioimage Data Loading for ML

**Question**: How file format and layout of bioimaging data affect ML training and inference throughput, locally and on S3

**GAP**: Existing benchmarks use generic read patterns on synthetic data. However, in model development data access is rarely straightforward such as random access during training.

## 1. Phase 1: 2D Microscopy: High Content Screening

> [!NOTE]<br>
> **Phase 1 status:**<br>
> ✅ **Completed:** data, conversion, verification<br> 
> 🚧 **In progress:** readers, workloads<br>
> ⏭️ **Next:** benchmark grid (local + S3)<br>

### 1.1. Data: See [LINK](DATA.md)

### 1.2. Variables

<table width="100%" border="1" style="border-collapse: collapse; text-align: left;">
  <thead>
    <tr>
      <th>Independent Variable</th>
      <th>Values</th>
      <th>Description</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td rowspan="4"><b>Format+Layout</b></td>
      <td>TIFF</td>
      <td>Tagged Image File Format, the long standing default for microscopy. Each file holds one 2D image.</td>
    </tr>
    <tr>
      <td>OME-Zarr v3 (unsharded)</td>
      <td>Zarr is a cloud native format for chunked N-dimensional arrays: each chunk id storage as its own file (or S3 object), with JSON metadata. Chunks are directly accessible, fast on cloud and easy to write in parallel. Very many small files, however. </td>
    </tr>
    <tr>
      <td>OME-Zarr v3 (sharded)</td>
      <td>Fewer files than unsharded because chunks are bundled into larger shard files with internal index.</td>
    </tr>
    <tr>
      <td>HDF5</td>
      <td>A single file containing chunked, compressed arrays organized in groups. Chunk lookups go through an internal index, so cloud reads need many requests.</td>
    </tr>
    <tr>
      <td rowspan="2"><b>Access Patterns</b></td>
      <td>Randomized (Training)</td>
      <td>Simulating an ML model data loader. Performance depends on per request cost and how much extra data each read pulls in.</td>
    </tr>
    <tr>
      <td>Sliding Window (Inference)</td>
      <td>Sweeps an overlapping window across every field of view in order. Performance depends on bulk throughput and how well reads can be prefetched ahead of a model.</td>
    </tr>
    <tr>
      <td rowspan="2"><b>Storage</b></td>
      <td>Local</td>
      <td>Data on the machine's internal drive. Requests are fast and cheap, so performance is limited mainly by decompression and Python overhead. Cold (OS cache cleared) and warm runs. </td>
    </tr>
    <tr>
      <td>S3</td>
      <td>Data in an Amazon S3 bucket (us-east-1), read over the network from a laptop. Every request carries tens of milliseconds of latency, so performance depends on how many requests a format needs and how well they run in parallel.</td>
    </tr>    
  </tbody>
</table>

**Swept**
- Simulated model step: 5-200 ms/batch

**Controlled**
- Chunks: 256x256
- zstd compression
- Patch size: 224
- Batch Size · DataLoader Workers · Library Versions · Hardware

## 1.3. Metrics

<table width="100%" border="1" style="border-collapse: collapse; text-align: left;">
  <thead>
    <tr>
      <th>Group</th>
      <th>Metrics</th>
      <th>Description</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td rowspan="4"><b>Throughput</b></td>
      <td>patches/sec</td>
      <td>How many patches the DataLoader delivers per second.</td>
    </tr>
    <tr>
      <td>Median Latency</td>
      <td>Time to load a single patch: the median</td>
    </tr>
    <tr>
      <td>p99 Latency</td>
      <td>Time to load a single patch: the slowest 1%</td>
    </tr>
    <tr>
      <td>Data-Wait Fraction</td>
      <td>Share of each step spent waiting for the next batch instead of computing/ Low values means GPU is fed, high values means GPU is starved by data loading.</td>
    </tr>
    <tr>
      <td rowspan="3"><b>Storage Cost</b></td>
      <td>File Count</td>
      <td>Number of files or objects needed to store the plate. Drives listing, copying, backup time, and HPC file quotas</td>
    </tr>
    <tr>
      <td>Size on Disk</td>
      <td>Total storage used, reflecting compression. Drives disk and S3 storage costs.</td>
    </tr>
    <tr>
      <td>Copy/Sync Time</td>
      <td>Time to copy data locally and to upload it to S3. Practical cost of moving data between systems.</td>
    </tr>    
  </tbody>
</table>

## 1.4. Fairness Rules
1. Every format is converted from the same source and checksum-verified against it.
2. Same chunk shape and codec in every format.
3. One read_patch(region) interface over all axes.
4. Seeded request lists, replayed identically for every format.
5. Cold runs clear the page cache first (sudo purge). Library versions and hardware are recorded.

## 1.5. Deliverables
1. Repo: installable with uv, reruns end to end.
2. Recommendation table: training/inference × local/S3, with speedup over TIFF and data-wait fraction per cell.
4. Figures: read-amplification explainer, local → S3 arrow plot, bottleneck curve.
5. README: limitations (one plate, one assay, 2D only, a single Mac, laptop-to-S3) and next phases.