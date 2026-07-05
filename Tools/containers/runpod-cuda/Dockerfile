# syntax=docker/dockerfile:1.7
#
# Build from the repository root:
#   docker build -f Tools/containers/runpod-cuda/Dockerfile -t warpx-cuda-runpod .
#
# This image installs CUDA-enabled WarpX binaries for all geometries:
#   warpx.1d, warpx.2d, warpx.3d, warpx.rz, warpx.rcylinder, warpx.rsphere

ARG CUDA_IMAGE_TAG=12.4.1

FROM nvidia/cuda:${CUDA_IMAGE_TAG}-devel-ubuntu22.04 AS build

ARG CMAKE_CUDA_ARCHITECTURES="75;80;86;89;90"
ARG NJOBS=8
ARG WARPX_DIMS="1;2;3;RZ;RCYLINDER;RSPHERE"

ENV DEBIAN_FRONTEND=noninteractive \
    CMAKE_GENERATOR=Ninja

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        cmake \
        git \
        libfftw3-dev \
        libfftw3-mpi-dev \
        libhdf5-mpi-dev \
        libopenmpi-dev \
        ninja-build \
        openmpi-bin \
        pkg-config \
        python3 && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /src/warpx
COPY . /src/warpx

RUN cmake \
        -S /src/warpx \
        -B /tmp/warpx-build \
        -DBUILD_SHARED_LIBS=ON \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_CUDA_ARCHITECTURES="${CMAKE_CUDA_ARCHITECTURES}" \
        -DCMAKE_INSTALL_PREFIX=/opt/warpx \
        -DWarpX_APP=ON \
        -DWarpX_COMPUTE=CUDA \
        -DWarpX_DIMS="${WARPX_DIMS}" \
        -DWarpX_FFT=ON \
        -DWarpX_MPI=ON \
        -DWarpX_OPENPMD=OFF \
        -DWarpX_PYTHON=OFF \
        -DWarpX_QED_TABLE_GEN=OFF && \
    cmake --build /tmp/warpx-build --target install -j "${NJOBS}" && \
    for geom in 1d 2d 3d rz rcylinder rsphere; do \
        target="$(find /opt/warpx/bin -maxdepth 1 -type f -name "warpx.${geom}.*" -print -quit)"; \
        test -n "${target}"; \
        ln -sf "${target}" "/opt/warpx/bin/warpx.${geom}"; \
    done && \
    rm -rf /tmp/warpx-build /opt/warpx/lib/*.a

FROM ghcr.io/astral-sh/uv:0.7.13 AS uv

FROM nvidia/cuda:${CUDA_IMAGE_TAG}-runtime-ubuntu22.04 AS runtime

ENV DEBIAN_FRONTEND=noninteractive \
    HDF5_USE_FILE_LOCKING=FALSE \
    LD_LIBRARY_PATH=/opt/warpx/lib:${LD_LIBRARY_PATH} \
    OMPI_ALLOW_RUN_AS_ROOT=1 \
    OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1 \
    PATH=/opt/warpx/bin:${PATH} \
    WARPX_INPUT_DIR=/work/inputs \
    WARPX_OUTPUT_DIR=/work/outputs \
    WARPX_ROOT=/opt/warpx

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        ca-certificates \
        libfftw3-double3 \
        libfftw3-mpi3 \
        libgomp1 \
        libhdf5-openmpi-103-1 \
        libopenmpi3 \
        openmpi-bin \
        python3-minimal && \
    rm -rf /var/lib/apt/lists/* && \
    mkdir -p /work/inputs /work/outputs

COPY --from=build /opt/warpx /opt/warpx
COPY --from=uv /uv /uvx /usr/local/bin/
COPY Tools/containers/runpod-cuda/entrypoint.sh /opt/warpx/bin/warpx-container

RUN chmod +x /opt/warpx/bin/warpx-container && \
    ln -sf /opt/warpx/bin/warpx-container /usr/local/bin/warpx-container

VOLUME ["/work/inputs", "/work/outputs"]
WORKDIR /work/outputs
ENTRYPOINT ["/opt/warpx/bin/warpx-container"]
CMD ["bash"]
