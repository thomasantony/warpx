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

add_warpx_test(
    test_3d_external_current_from_m_rmf  # name
    3  # dims
    2  # nprocs
    inputs_test_3d_external_current_from_m_rmf  # inputs
    "analysis_rmf.py diags/diag000001 diags/diag000002"  # analysis
    OFF  # checksum
    OFF  # dependency
)
