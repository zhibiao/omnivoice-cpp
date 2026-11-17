cmake_minimum_required(VERSION 3.20)

project(omnivoice_cpp VERSION 0.1.0 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

set(BUILD_SHARED_LIBS ON CACHE BOOL "Build shared GGML backend libraries")

if(BUILD_SHARED_LIBS)
    set(GGML_BACKEND_DL ON CACHE BOOL "" FORCE)
    set(GGML_NATIVE OFF CACHE BOOL "" FORCE)
    set(GGML_CPU_ALL_VARIANTS ON CACHE BOOL "" FORCE)
else()
    set(GGML_BACKEND_DL OFF CACHE BOOL "" FORCE)
endif()

set(GGML_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(GGML_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(GGML_CCACHE OFF CACHE BOOL "" FORCE)

option(
    OMNIVOICE_ENABLE_CUDA
    "Build OmniVoice with the GGML CUDA backend"
    ON
)

set(
    OMNIVOICE_CUDA_ARCHITECTURES
    ""
    CACHE STRING
    "CUDA architectures: -real = SASS, -virtual = PTX"
)

if(OMNIVOICE_ENABLE_CUDA)
    set(GGML_CUDA ON CACHE BOOL "" FORCE)
    set(GGML_CUDA_NCCL OFF CACHE BOOL "" FORCE)
    set(GGML_CUDA_GRAPHS ON CACHE BOOL "" FORCE)
    set(
        CMAKE_CUDA_ARCHITECTURES
        "${OMNIVOICE_CUDA_ARCHITECTURES}"
        CACHE STRING "" FORCE
    )
    message(STATUS "OmniVoice CUDA backend enabled")
    message(
        STATUS
        "OmniVoice CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}"
    )
else()
    set(GGML_CUDA OFF CACHE BOOL "" FORCE)
    set(GGML_CUDA_GRAPHS OFF CACHE BOOL "" FORCE)
    message(STATUS "OmniVoice CUDA backend disabled")
endif()

if(POLICY CMP0194)
    cmake_policy(SET CMP0194 NEW)
endif()

add_subdirectory(vendor/ggml-0.29.0)

function(omnivoice_configure_target target)
    if(MSVC)
        target_compile_options(${target} PRIVATE /utf-8)
    endif()
endfunction()

add_library(omnivoice-core STATIC
    src/audio_postprocess.cpp
    src/audio_io.cpp
    src/config.cpp
    src/diffusion_decoder/diffusion_decoder.cpp
    src/diffusion_decoder/mask_schedule.cpp
    src/diffusion_decoder/token_selector.cpp
    src/duration_estimator.cpp
    src/gguf_model.cpp
    src/higgs_tokenizer/decoder.cpp
    src/higgs_tokenizer/encoder.cpp
    src/higgs_tokenizer/resampler.cpp
    src/omnivoice.cpp
    src/prompt_builder.cpp
    src/qwen3_lm/qwen3_block.cpp
    src/qwen3_lm/qwen3_model.cpp
    src/text_tokenizer.cpp
    src/text_chunker.cpp
    src/voice_clone_prompt.cpp
    src/wav_writer.cpp
)
omnivoice_configure_target(omnivoice-core)
target_include_directories(omnivoice-core PUBLIC
    ${PROJECT_SOURCE_DIR}/src
)
target_link_libraries(omnivoice-core PUBLIC ggml)
if(WIN32)
    target_link_libraries(omnivoice-core PRIVATE normaliz)
endif()

add_executable(omnivoice-cli
    app/omnivoice_cli.cpp
)
omnivoice_configure_target(omnivoice-cli)
target_link_libraries(omnivoice-cli PRIVATE omnivoice-core)
target_compile_definitions(omnivoice-cli PRIVATE
    OMNIVOICE_VERSION="${PROJECT_VERSION}"
)

if(WIN32)
    target_link_libraries(omnivoice-cli PRIVATE shell32)
endif()

include(CTest)

if(BUILD_TESTING)
    add_subdirectory(tests)
endif()
