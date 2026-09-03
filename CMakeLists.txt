cmake_minimum_required(VERSION 3.20...4.4)

if(DEFINED SKBUILD_PROJECT_NAME)
    project(${SKBUILD_PROJECT_NAME} VERSION ${SKBUILD_PROJECT_VERSION} LANGUAGES CXX)
else()
    project(qupy VERSION 0.3.0 LANGUAGES CXX)
endif()

option(QUPY_BUILD_PYTHON "Build the Python extension" ON)
option(QUPY_BUILD_TESTS "Build native C++ tests" OFF)
option(QUPY_WARNINGS_AS_ERRORS "Treat native compiler warnings as errors" OFF)
option(QUPY_ENABLE_SANITIZERS "Enable AddressSanitizer and UndefinedBehaviorSanitizer" OFF)
option(QUPY_ENABLE_MPI "Enable MPI distributed execution when an MPI runtime is available" ON)

add_library(
    qupy_core STATIC
    src/cpp/core.cpp
    src/cpp/advanced.cpp
    src/cpp/circuit.cpp
    src/cpp/compiler.cpp
    src/cpp/cuda_driver.cpp
    src/cpp/distributed.cpp
    src/cpp/fingerprint.cpp
    src/cpp/mps.cpp
    src/cpp/multi_device.cpp
    src/cpp/provider.cpp
    src/cpp/stabilizer.cpp
    src/cpp/tensor_network.cpp
    src/cpp/trajectory.cpp
)
target_compile_features(qupy_core PUBLIC cxx_std_20)
target_include_directories(
    qupy_core
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)
set_target_properties(qupy_core PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_link_libraries(qupy_core PRIVATE ${CMAKE_DL_LIBS})

if(MSVC)
    target_compile_options(qupy_core PRIVATE /W4 /permissive-)
    if(QUPY_WARNINGS_AS_ERRORS)
        target_compile_options(qupy_core PRIVATE /WX)
    endif()
else()
    target_compile_options(qupy_core PRIVATE -Wall -Wextra -Wpedantic)
    if(QUPY_WARNINGS_AS_ERRORS)
        target_compile_options(qupy_core PRIVATE -Werror)
    endif()
    if(QUPY_ENABLE_SANITIZERS)
        target_compile_options(qupy_core PRIVATE -fsanitize=address,undefined -fno-omit-frame-pointer)
        target_compile_definitions(qupy_core PUBLIC QUPY_SANITIZER_BUILD=1)
    endif()
endif()

find_package(OpenMP QUIET)
if(OpenMP_CXX_FOUND)
    target_link_libraries(qupy_core PUBLIC OpenMP::OpenMP_CXX)
    target_compile_definitions(qupy_core PUBLIC QUPY_HAS_OPENMP=1)
endif()

if(QUPY_ENABLE_MPI)
    find_package(MPI QUIET COMPONENTS CXX)
    if(MPI_CXX_FOUND)
        target_link_libraries(qupy_core PUBLIC MPI::MPI_CXX)
        target_compile_definitions(qupy_core PUBLIC QUPY_HAS_MPI=1)
    endif()
endif()

if(QUPY_BUILD_PYTHON)
    find_package(Python 3.12 REQUIRED COMPONENTS Interpreter Development.Module)
    find_package(nanobind CONFIG REQUIRED)

    nanobind_add_module(
        _native NB_STATIC
        src/cpp/bindings.cpp
        src/cpp/advanced_bindings.cpp
        src/cpp/circuit_bindings.cpp
        src/cpp/compiler_bindings.cpp
        src/cpp/multi_device_bindings.cpp
        src/cpp/tensor_network_bindings.cpp
        src/cpp/trajectory_bindings.cpp
    )
    target_compile_features(_native PRIVATE cxx_std_20)
    target_link_libraries(_native PRIVATE qupy_core)
    if(MSVC)
        target_compile_options(_native PRIVATE /W4 /permissive-)
    else()
        target_compile_options(_native PRIVATE -Wall -Wextra -Wpedantic)
    endif()
    install(TARGETS _native LIBRARY DESTINATION qupy RUNTIME DESTINATION qupy)
endif()

if(QUPY_BUILD_TESTS)
    enable_testing()
    add_executable(qupy_core_tests tests/cpp/test_core.cpp)
    target_compile_features(qupy_core_tests PRIVATE cxx_std_20)
    target_link_libraries(qupy_core_tests PRIVATE qupy_core)
    target_include_directories(qupy_core_tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src/cpp)
    if(MSVC)
        target_compile_options(qupy_core_tests PRIVATE /W4 /permissive-)
        if(QUPY_WARNINGS_AS_ERRORS)
            target_compile_options(qupy_core_tests PRIVATE /WX)
        endif()
    else()
        target_compile_options(qupy_core_tests PRIVATE -Wall -Wextra -Wpedantic)
        if(QUPY_WARNINGS_AS_ERRORS)
            target_compile_options(qupy_core_tests PRIVATE -Werror)
        endif()
        if(QUPY_ENABLE_SANITIZERS)
            target_compile_options(qupy_core_tests PRIVATE -fsanitize=address,undefined -fno-omit-frame-pointer)
            target_link_options(qupy_core_tests PRIVATE -fsanitize=address,undefined)
        endif()
    endif()
    add_test(NAME qupy_core_tests COMMAND qupy_core_tests)

    add_executable(qupy_circuit_tests tests/cpp/test_circuit.cpp)
    target_compile_features(qupy_circuit_tests PRIVATE cxx_std_20)
    target_link_libraries(qupy_circuit_tests PRIVATE qupy_core)
    if(MSVC)
        target_compile_options(qupy_circuit_tests PRIVATE /W4 /permissive-)
        if(QUPY_WARNINGS_AS_ERRORS)
            target_compile_options(qupy_circuit_tests PRIVATE /WX)
        endif()
    else()
        target_compile_options(qupy_circuit_tests PRIVATE -Wall -Wextra -Wpedantic)
        if(QUPY_WARNINGS_AS_ERRORS)
            target_compile_options(qupy_circuit_tests PRIVATE -Werror)
        endif()
        if(QUPY_ENABLE_SANITIZERS)
            target_compile_options(
                qupy_circuit_tests PRIVATE -fsanitize=address,undefined -fno-omit-frame-pointer
            )
            target_link_options(qupy_circuit_tests PRIVATE -fsanitize=address,undefined)
        endif()
    endif()
    add_test(NAME qupy_circuit_tests COMMAND qupy_circuit_tests)

    add_executable(qupy_compiler_tests tests/cpp/test_compiler.cpp)
    target_compile_features(qupy_compiler_tests PRIVATE cxx_std_20)
    target_link_libraries(qupy_compiler_tests PRIVATE qupy_core)
    if(MSVC)
        target_compile_options(qupy_compiler_tests PRIVATE /W4 /permissive-)
        if(QUPY_WARNINGS_AS_ERRORS)
            target_compile_options(qupy_compiler_tests PRIVATE /WX)
        endif()
    else()
        target_compile_options(qupy_compiler_tests PRIVATE -Wall -Wextra -Wpedantic)
        if(QUPY_WARNINGS_AS_ERRORS)
            target_compile_options(qupy_compiler_tests PRIVATE -Werror)
        endif()
        if(QUPY_ENABLE_SANITIZERS)
            target_compile_options(
                qupy_compiler_tests PRIVATE -fsanitize=address,undefined -fno-omit-frame-pointer
            )
            target_link_options(qupy_compiler_tests PRIVATE -fsanitize=address,undefined)
        endif()
    endif()
    add_test(NAME qupy_compiler_tests COMMAND qupy_compiler_tests)

    add_executable(qupy_tensor_network_tests tests/cpp/test_tensor_network.cpp)
    target_compile_features(qupy_tensor_network_tests PRIVATE cxx_std_20)
    target_link_libraries(qupy_tensor_network_tests PRIVATE qupy_core)
    if(MSVC)
        target_compile_options(qupy_tensor_network_tests PRIVATE /W4 /permissive-)
        if(QUPY_WARNINGS_AS_ERRORS)
            target_compile_options(qupy_tensor_network_tests PRIVATE /WX)
        endif()
    else()
        target_compile_options(qupy_tensor_network_tests PRIVATE -Wall -Wextra -Wpedantic)
        if(QUPY_WARNINGS_AS_ERRORS)
            target_compile_options(qupy_tensor_network_tests PRIVATE -Werror)
        endif()
        if(QUPY_ENABLE_SANITIZERS)
            target_compile_options(
                qupy_tensor_network_tests PRIVATE -fsanitize=address,undefined -fno-omit-frame-pointer
            )
            target_link_options(qupy_tensor_network_tests PRIVATE -fsanitize=address,undefined)
        endif()
    endif()
    add_test(NAME qupy_tensor_network_tests COMMAND qupy_tensor_network_tests)

    add_executable(qupy_trajectory_tests tests/cpp/test_trajectory.cpp)
    target_compile_features(qupy_trajectory_tests PRIVATE cxx_std_20)
    target_link_libraries(qupy_trajectory_tests PRIVATE qupy_core)
    if(MSVC)
        target_compile_options(qupy_trajectory_tests PRIVATE /W4 /permissive-)
        if(QUPY_WARNINGS_AS_ERRORS)
            target_compile_options(qupy_trajectory_tests PRIVATE /WX)
        endif()
    else()
        target_compile_options(qupy_trajectory_tests PRIVATE -Wall -Wextra -Wpedantic)
        if(QUPY_WARNINGS_AS_ERRORS)
            target_compile_options(qupy_trajectory_tests PRIVATE -Werror)
        endif()
        if(QUPY_ENABLE_SANITIZERS)
            target_compile_options(
                qupy_trajectory_tests PRIVATE -fsanitize=address,undefined -fno-omit-frame-pointer
            )
            target_link_options(qupy_trajectory_tests PRIVATE -fsanitize=address,undefined)
        endif()
    endif()
    add_test(NAME qupy_trajectory_tests COMMAND qupy_trajectory_tests)

    add_library(qupy_test_provider SHARED tests/cpp/provider_fixture.cpp)
    target_compile_features(qupy_test_provider PRIVATE cxx_std_20)
    target_include_directories(qupy_test_provider PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)

    add_executable(qupy_advanced_tests tests/cpp/test_advanced.cpp)
    target_compile_features(qupy_advanced_tests PRIVATE cxx_std_20)
    target_link_libraries(qupy_advanced_tests PRIVATE qupy_core)
    target_compile_definitions(
        qupy_advanced_tests PRIVATE
        QUPY_TEST_PROVIDER_PATH="$<TARGET_FILE:qupy_test_provider>"
    )
    add_dependencies(qupy_advanced_tests qupy_test_provider)

    if(MSVC)
        target_compile_options(qupy_test_provider PRIVATE /W4 /permissive-)
        target_compile_options(qupy_advanced_tests PRIVATE /W4 /permissive-)
        if(QUPY_WARNINGS_AS_ERRORS)
            target_compile_options(qupy_test_provider PRIVATE /WX)
            target_compile_options(qupy_advanced_tests PRIVATE /WX)
        endif()
    else()
        target_compile_options(qupy_test_provider PRIVATE -Wall -Wextra -Wpedantic)
        target_compile_options(qupy_advanced_tests PRIVATE -Wall -Wextra -Wpedantic)
        if(QUPY_WARNINGS_AS_ERRORS)
            target_compile_options(qupy_test_provider PRIVATE -Werror)
            target_compile_options(qupy_advanced_tests PRIVATE -Werror)
        endif()
        if(QUPY_ENABLE_SANITIZERS)
            target_compile_options(qupy_advanced_tests PRIVATE -fsanitize=address,undefined -fno-omit-frame-pointer)
            target_link_options(qupy_advanced_tests PRIVATE -fsanitize=address,undefined)
        endif()
    endif()
    add_test(NAME qupy_advanced_tests COMMAND qupy_advanced_tests)

    if(MPI_CXX_FOUND)
        add_executable(qupy_mpi_tests tests/cpp/test_mpi.cpp)
        target_compile_features(qupy_mpi_tests PRIVATE cxx_std_20)
        target_link_libraries(qupy_mpi_tests PRIVATE qupy_core)
        if(MSVC)
            target_compile_options(qupy_mpi_tests PRIVATE /W4 /permissive-)
            if(QUPY_WARNINGS_AS_ERRORS)
                target_compile_options(qupy_mpi_tests PRIVATE /WX)
            endif()
        else()
            target_compile_options(qupy_mpi_tests PRIVATE -Wall -Wextra -Wpedantic)
            if(QUPY_WARNINGS_AS_ERRORS)
                target_compile_options(qupy_mpi_tests PRIVATE -Werror)
            endif()
            if(QUPY_ENABLE_SANITIZERS)
                target_compile_options(qupy_mpi_tests PRIVATE -fsanitize=address,undefined -fno-omit-frame-pointer)
                target_link_options(qupy_mpi_tests PRIVATE -fsanitize=address,undefined)
            endif()
        endif()
        add_test(
            NAME qupy_mpi_2
            COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 2
                    ${MPIEXEC_PREFLAGS} $<TARGET_FILE:qupy_mpi_tests> ${MPIEXEC_POSTFLAGS}
        )
        add_test(
            NAME qupy_mpi_4
            COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 4
                    ${MPIEXEC_PREFLAGS} $<TARGET_FILE:qupy_mpi_tests> ${MPIEXEC_POSTFLAGS}
        )
    endif()
endif()
