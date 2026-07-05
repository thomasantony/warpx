#!/usr/bin/env bash
set -euo pipefail

: "${WARPX_INPUT_DIR:=/work/inputs}"
: "${WARPX_OUTPUT_DIR:=/work/outputs}"

sync_results() {
    if [[ -z "${WARPX_S3_URI:-}" ]]; then
        echo "WARPX_S3_URI is not set; skipping result sync." >&2
        return 0
    fi

    local endpoint_args=()
    if [[ -n "${AWS_ENDPOINT_URL:-}" ]]; then
        endpoint_args=(--endpoint-url "${AWS_ENDPOINT_URL}")
    fi

    uvx --from awscli aws s3 sync \
        "${WARPX_OUTPUT_DIR}/" \
        "${WARPX_S3_URI}" \
        "${endpoint_args[@]}"
}

run_warpx() {
    local geometry="${1:-}"
    local input_file="${2:-}"

    if [[ -z "${geometry}" || -z "${input_file}" ]]; then
        echo "usage: warpx-container run <1d|2d|3d|rz|rcylinder|rsphere> <input-file> [warpx args...]" >&2
        return 64
    fi

    shift 2

    local executable="warpx.${geometry}"
    local input_path="${input_file}"
    if [[ "${input_file}" != /* ]]; then
        input_path="${WARPX_INPUT_DIR}/${input_file}"
    fi

    mkdir -p "${WARPX_OUTPUT_DIR}"
    cd "${WARPX_OUTPUT_DIR}"

    "${executable}" "${input_path}" "$@"
}

if [[ $# -eq 0 ]]; then
    set -- bash
fi

case "${1}" in
    run)
        shift
        set +e
        run_warpx "$@"
        status=$?
        set -e
        if [[ "${status}" -eq 0 && "${WARPX_SYNC_RESULTS:-0}" == "1" ]]; then
            sync_results
        fi
        exit "${status}"
        ;;
    sync-results)
        sync_results
        ;;
    bash|sh|warpx.*|mpirun|mpiexec|uv|uvx)
        exec "$@"
        ;;
    *)
        exec "$@"
        ;;
esac
