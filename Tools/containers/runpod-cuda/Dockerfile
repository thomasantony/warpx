# syntax=docker/dockerfile:1.7
#
# Build from the repository root:
#   docker build -f Tools/containers/runpod-cuda/Dockerfile -t warpx-cuda-runpod .
#
# This image installs CUDA-enabled WarpX binaries for all geometries:
#   warpx.1d, warpx.2d, warpx.3d, warpx.rz, warpx.rcylinder, warpx.rsphere

ARG CUDA_IMAGE_TAG=12.8.1

FROM nvidia/cuda:${CUDA_IMAGE_TAG}-devel-ubuntu22.04 AS build

ARG AMREX_CUDA_ARCH="8.9+PTX"
ARG CMAKE_CUDA_ARCHITECTURES="89-real;89-virtual"
ARG NJOBS
ARG WARPX_DIMS="1;2;3;RZ;RCYLINDER;RSPHERE"

ENV DEBIAN_FRONTEND=noninteractive \
    CMAKE_GENERATOR=Ninja

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        git \
        libblas-dev \
        libfftw3-dev \
        libfftw3-mpi-dev \
        libhdf5-mpi-dev \
        liblapack-dev \
        libopenmpi-dev \
        ninja-build \
        openmpi-bin \
        pkg-config \
        python3 \
        python3-pip && \
    python3 -m pip install --no-cache-dir "cmake>=3.24,<4" && \
    rm -rf /var/lib/apt/lists/*

RUN git clone --depth 1 https://github.com/icl-utk-edu/blaspp.git /tmp/blaspp && \
    cmake \
        -S /tmp/blaspp \
        -B /tmp/blaspp-build \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_CUDA_ARCHITECTURES="${CMAKE_CUDA_ARCHITECTURES}" \
        -DCMAKE_CXX_STANDARD=20 \
        -DCMAKE_INSTALL_PREFIX=/opt/warpx \
        -Dbuild_tests=OFF \
        -Dgpu_backend=cuda \
        -Duse_openmp=OFF && \
    cmake --build /tmp/blaspp-build --target install -j "${NJOBS:-$(nproc)}" && \
    git clone --depth 1 https://github.com/icl-utk-edu/lapackpp.git /tmp/lapackpp && \
    cmake \
        -S /tmp/lapackpp \
        -B /tmp/lapackpp-build \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_CUDA_ARCHITECTURES="${CMAKE_CUDA_ARCHITECTURES}" \
        -DCMAKE_CXX_STANDARD=20 \
        -DCMAKE_INSTALL_PREFIX=/opt/warpx \
        -DCMAKE_INSTALL_RPATH_USE_LINK_PATH=ON \
        -DCMAKE_PREFIX_PATH=/opt/warpx \
        -Dbuild_tests=OFF \
        -Dgpu_backend=cuda \
        -Duse_cmake_find_lapack=ON && \
    cmake --build /tmp/lapackpp-build --target install -j "${NJOBS:-$(nproc)}" && \
    rm -rf /tmp/blaspp /tmp/blaspp-build /tmp/lapackpp /tmp/lapackpp-build

WORKDIR /src/warpx
COPY . /src/warpx

RUN cmake \
        -S /src/warpx \
        -B /tmp/warpx-build \
        -DBUILD_SHARED_LIBS=ON \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_CUDA_ARCHITECTURES="${CMAKE_CUDA_ARCHITECTURES}" \
        -DCMAKE_INSTALL_PREFIX=/opt/warpx \
        -DCMAKE_PREFIX_PATH=/opt/warpx \
        -DWarpX_APP=ON \
        -DAMReX_CUDA_ARCH="${AMREX_CUDA_ARCH}" \
        -DWarpX_COMPUTE=CUDA \
        -DWarpX_DIMS="${WARPX_DIMS}" \
        -DWarpX_FFT=ON \
        -DWarpX_MPI=ON \
        -DWarpX_OPENPMD=OFF \
        -DWarpX_PYTHON=OFF \
        -DWarpX_QED_TABLE_GEN=OFF && \
    cmake --build /tmp/warpx-build --target install -j "${NJOBS:-$(nproc)}" && \
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
        libblas3 \
        libfftw3-double3 \
        libfftw3-mpi3 \
        libgomp1 \
        libhdf5-openmpi-103-1 \
        liblapack3 \
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
