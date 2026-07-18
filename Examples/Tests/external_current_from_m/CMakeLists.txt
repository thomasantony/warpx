# Add tests (alphabetical order) ##############################################
#

add_warpx_test(
    test_3d_external_current_from_m  # name
    3  # dims
    2  # nprocs
    inputs_test_3d_external_current_from_m  # inputs
    "analysis.py diags/diag000001"  # analysis
    OFF  # checksum
    OFF  # dependency
)
