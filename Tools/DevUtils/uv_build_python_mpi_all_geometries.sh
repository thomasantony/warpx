#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${repo_root}"

export PIP_CACHE_DIR="${repo_root}/.uv-cache/pip"

build_dir="${WARPX_UV_BUILD_DIR:-build-uv}"
dims="${WARPX_DIMS:-1;2;3;RZ;RCYLINDER;RSPHERE}"
jobs="${BUILD_PARALLEL:-$(sysctl -n hw.logicalcpu 2>/dev/null || echo 4)}"
generator="${CMAKE_GENERATOR:-Ninja}"

amrex_lib_dir="${repo_root}/${build_dir}/lib"
pyamrex_src_dir="${repo_root}/${build_dir}/_deps/fetchedpyamrex-src"
pyamrex_build_dir="${repo_root}/${build_dir}/_deps/fetchedpyamrex-build"
pyamrex_lib_dir="${repo_root}/${build_dir}/lib/site-packages/amrex"
pywarpx_lib_dir="${repo_root}/${build_dir}/lib/site-packages/pywarpx"

build_targets() {
    local target
    for target in "$@"; do
        uv run --no-sync cmake --build "${build_dir}" --target "${target}" -j "${jobs}"
    done
}

copy_amrex_dylibs() {
    cp "${amrex_lib_dir}/libamrex_1d.dylib" "$1/libamrex_1d.dylib"
    cp "${amrex_lib_dir}/libamrex_2d.dylib" "$2/libamrex_2d.dylib"
    cp "${amrex_lib_dir}/libamrex_3d.dylib" "$3/libamrex_3d.dylib"
}

fix_darwin_rpath() {
    [[ "$(uname -s)" == "Darwin" ]] || return 0

    local module
    for module in "$@"; do
        if otool -l "${module}" | grep -q "path ${amrex_lib_dir} "; then
            install_name_tool -delete_rpath "${amrex_lib_dir}" "${module}"
        fi
        if ! otool -l "${module}" | grep -q "path @loader_path "; then
            install_name_tool -add_rpath "@loader_path" "${module}"
        fi
    done
}

patch_pyamrex_binary_wheel() {
    local setup_file="${pyamrex_src_dir}/setup.py"

    if grep -q "class BinaryWheel" "${setup_file}"; then
        return
    fi

    uv run --no-sync python - "${setup_file}" <<'PY'
import pathlib
import sys

setup_file = pathlib.Path(sys.argv[1])
text = setup_file.read_text()
text = text.replace(
    "from setuptools.command.build_ext import build_ext\n",
    """from setuptools.command.build_ext import build_ext

try:
    from setuptools.command.bdist_wheel import bdist_wheel
except ImportError:
    try:
        from wheel.bdist_wheel import bdist_wheel
    except ImportError:
        bdist_wheel = None
""",
)
text = text.replace(
    "\n\nclass CMakeExtension(Extension):\n",
    """

if bdist_wheel is not None:

    class BinaryWheel(bdist_wheel):
        def finalize_options(self):
            bdist_wheel.finalize_options(self)
            self.root_is_pure = False


class CMakeExtension(Extension):
""",
)
text = text.replace(
    "if PYAMREX_libdir:\n    cmdclass = dict(build=CopyPreBuild)\n",
    """if PYAMREX_libdir:
    cmdclass = dict(build=CopyPreBuild)
    if bdist_wheel is not None:
        cmdclass["bdist_wheel"] = BinaryWheel
""",
)
setup_file.write_text(text)
PY
}

configure_build() {
    uv sync --no-install-project

    local python_exe
    python_exe="$(uv run --no-sync python -c 'import sys; print(sys.executable)')"

    local cmake_args=(
        -S .
        -B "${build_dir}"
        -G "${generator}"
        "-DPython_EXECUTABLE=${python_exe}"
        "-DPython_ROOT_DIR=$(dirname "$(dirname "${python_exe}")")"
        "-DWarpX_DIMS=${dims}"
        -DWarpX_PYTHON=ON
        -DWarpX_MPI=ON
        -DWarpX_COMPUTE=OMP
        -DWarpX_CCACHE=OFF
        -DWarpX_PYTHON_IPO=OFF
        -DpyAMReX_IPO=OFF
    )

    if [[ "${WARPX_CMAKE_FRESH:-OFF}" == "ON" || ! -f "${build_dir}/CMakeCache.txt" ]]; then
        cmake_args=(--fresh "${cmake_args[@]}")
    fi

    if [[ "$(uname -s)" == "Darwin" ]]; then
        local libomp_prefix
        libomp_prefix="${LIBOMP_ROOT:-$(brew --prefix libomp 2>/dev/null || true)}"
        if [[ -n "${libomp_prefix}" && -f "${libomp_prefix}/lib/libomp.dylib" ]]; then
            cmake_args+=(
                "-DOpenMP_C_FLAGS=-Xpreprocessor -fopenmp -I${libomp_prefix}/include"
                "-DOpenMP_CXX_FLAGS=-Xpreprocessor -fopenmp -I${libomp_prefix}/include"
                -DOpenMP_C_LIB_NAMES=omp
                -DOpenMP_CXX_LIB_NAMES=omp
                "-DOpenMP_omp_LIBRARY=${libomp_prefix}/lib/libomp.dylib"
                "-DCMAKE_EXE_LINKER_FLAGS=-L${libomp_prefix}/lib -Wl,-rpath,${libomp_prefix}/lib"
                "-DCMAKE_SHARED_LINKER_FLAGS=-L${libomp_prefix}/lib -Wl,-rpath,${libomp_prefix}/lib"
            )
        fi
    fi

    uv run --no-sync cmake "${cmake_args[@]}"
}

build_amrex_wheel() {
    build_targets pyAMReX_1d pyAMReX_2d pyAMReX_3d pyAMReX_python_sources
    copy_amrex_dylibs \
        "${pyamrex_lib_dir}/space1d" \
        "${pyamrex_lib_dir}/space2d" \
        "${pyamrex_lib_dir}/space3d"
    fix_darwin_rpath "${pyamrex_lib_dir}"/space*d/amrex_*_pybind*.so
    patch_pyamrex_binary_wheel

    rm -rf "${pyamrex_build_dir}/amrex-whl"
    (
        cd "${pyamrex_build_dir}"
        PYAMREX_LIBDIR="${pyamrex_lib_dir}" uv run --no-sync python -m pip -v wheel \
            --no-build-isolation --no-deps --wheel-dir=amrex-whl "${pyamrex_src_dir}"
    )
}

build_pywarpx_wheel() {
    build_targets \
        pyWarpX_1d pyWarpX_2d pyWarpX_3d \
        pyWarpX_rz pyWarpX_rcylinder pyWarpX_rsphere \
        pyWarpX_python_sources
    copy_amrex_dylibs "${pywarpx_lib_dir}" "${pywarpx_lib_dir}" "${pywarpx_lib_dir}"
    fix_darwin_rpath "${pywarpx_lib_dir}"/warpx_pybind_*.so

    rm -rf "${build_dir}/warpx-whl"
    (
        cd "${build_dir}"
        PYWARPX_LIB_DIR="${pywarpx_lib_dir}" uv run --no-sync python -m pip -v wheel \
            --no-build-isolation --no-deps --wheel-dir=warpx-whl "${repo_root}"
    )
}

install_wheels() {
    mkdir -p dist
    rm -f dist/amrex-*.whl dist/pywarpx-*.whl
    cp "${pyamrex_build_dir}"/amrex-whl/*.whl dist/
    cp "${build_dir}"/warpx-whl/*.whl dist/
    uv run --no-sync python -m pip install --force-reinstall --no-deps dist/amrex-*.whl dist/pywarpx-*.whl
    ls -lh dist/*.whl
}

configure_build
build_amrex_wheel
build_pywarpx_wheel
install_wheels
