if(WarpX_MPI)
    set(native_yee_nprocs 2)
else()
    set(native_yee_nprocs 1)
endif()

add_warpx_test(
    test_3d_native_yee_energy_balance
    3
    ${native_yee_nprocs}
    inputs_test_3d_native_yee_energy_balance
    "analysis.py"
    OFF  # The test checks identities, not a field checksum.
    OFF
)

if(TEST test_3d_native_yee_energy_balance.run)
    # Production segments refuse to overwrite CSV evidence. Repeated CTest
    # invocations may remove this one output in their own dedicated run folder.
    add_test(
        NAME test_3d_native_yee_energy_balance.prepare
        COMMAND ${CMAKE_COMMAND} -E rm -f diags/native_yee_energy_balance.csv
        WORKING_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/test_3d_native_yee_energy_balance
    )
    set_tests_properties(test_3d_native_yee_energy_balance.prepare PROPERTIES
        FIXTURES_SETUP native_yee_csv_ready)
    set_tests_properties(test_3d_native_yee_energy_balance.run PROPERTIES
        FIXTURES_REQUIRED native_yee_csv_ready
        TIMEOUT 120)
endif()
