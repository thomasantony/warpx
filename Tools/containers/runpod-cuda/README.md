# WarpX CUDA RunPod Container

This container builds CUDA-enabled WarpX binaries for all geometries and keeps
the runtime image focused on running simulations from mounted input and output
directories.

## Build

Run from the repository root:

```bash
docker build -f Tools/containers/runpod-cuda/Dockerfile -t warpx-cuda-runpod .
```

Useful build arguments:

```bash
docker build -f Tools/containers/runpod-cuda/Dockerfile \
    --build-arg NJOBS=16 \
    --build-arg AMREX_CUDA_ARCH="8.0;8.6;8.9+PTX" \
    --build-arg CMAKE_CUDA_ARCHITECTURES="75;80;86;89;90" \
    -t warpx-cuda-runpod .
```

The default CUDA architectures cover common RunPod GPUs such as T4, A40, A100,
RTX 4090/L40, and H100. Narrowing the list makes the binaries smaller.

## Run

Mount input files at `/work/inputs` and collect outputs from `/work/outputs`:

```bash
docker run --rm --gpus all \
    -v "$PWD/inputs:/work/inputs:ro" \
    -v "$PWD/results:/work/outputs" \
    warpx-cuda-runpod \
    run 3d inputs
```

All geometry executables are installed:

```bash
warpx.1d
warpx.2d
warpx.3d
warpx.rz
warpx.rcylinder
warpx.rsphere
```

For multi-rank runs inside one container, use `mpirun` directly:

```bash
docker run --rm --gpus all \
    -v "$PWD/inputs:/work/inputs:ro" \
    -v "$PWD/results:/work/outputs" \
    warpx-cuda-runpod \
    mpirun -np 2 warpx.3d /work/inputs/inputs
```

RunPod caveats:

- Select a CUDA-capable pod image/runtime and expose GPUs to Docker.
- Most RunPod templates run as root; Open MPI root execution is enabled in the
  image for that reason.
- Use persistent volumes or network storage for `/work/outputs`; ephemeral pod
  disks can disappear when the pod is stopped.
- The MPI in this image is intended for single-container runs. Multi-node cloud
  MPI needs provider-specific networking and launch configuration.

## GPU mapped-memory compatibility

Some GPU container runtimes do not support device access to mapped pinned host
memory. Enable explicit device staging for the affected AMReX operations with:

```text
amrex.reduce_use_device_result = 1
```

This keeps pinned allocations host-only. GPU kernels write temporary device
storage, and AMReX explicitly copies each result back to its pinned host
destination. The setting does not change the main AMReX arena or require CUDA
managed-memory support.

## Optional S3/R2 Uploads

The runtime includes `uv`, so the AWS CLI can be run on demand with `uvx` instead
of being permanently installed in the image.

To sync results manually:

```bash
docker run --rm --gpus all \
    -e AWS_ACCESS_KEY_ID \
    -e AWS_SECRET_ACCESS_KEY \
    -e AWS_DEFAULT_REGION=auto \
    -e AWS_ENDPOINT_URL=https://<account-id>.r2.cloudflarestorage.com \
    -e WARPX_S3_URI=s3://<bucket>/<prefix> \
    -v "$PWD/results:/work/outputs" \
    warpx-cuda-runpod \
    sync-results
```

To sync automatically after a successful `run` command, add
`-e WARPX_SYNC_RESULTS=1` and the same S3/R2 environment variables.
