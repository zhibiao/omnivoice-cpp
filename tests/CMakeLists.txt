set(
    OMNIVOICE_TEST_DATA_DIR
    "${PROJECT_SOURCE_DIR}/testdata"
    CACHE PATH
    "Directory containing generated golden test data"
)
set(
    OMNIVOICE_MODEL_DIR
    "${PROJECT_SOURCE_DIR}/models"
    CACHE PATH
    "Directory containing converted GGUF models"
)
file(TO_CMAKE_PATH "${OMNIVOICE_TEST_DATA_DIR}" OMNIVOICE_TEST_DATA_DIR)
file(TO_CMAKE_PATH "${OMNIVOICE_MODEL_DIR}" OMNIVOICE_MODEL_DIR)

function(omnivoice_add_test target test_name)
    add_executable(${target} ${target}.cpp)
    omnivoice_configure_target(${target})
    target_link_libraries(${target} PRIVATE omnivoice-core)
    target_include_directories(${target} PRIVATE
        ${CMAKE_SOURCE_DIR}/vendor
    )
    target_compile_definitions(${target} PRIVATE
        OMNIVOICE_TEST_DATA_DIR="${OMNIVOICE_TEST_DATA_DIR}"
    )
    add_test(NAME ${test_name} COMMAND ${target})
    set_tests_properties(${test_name} PROPERTIES
        LABELS "requires-assets"
    )
endfunction()

function(omnivoice_add_cuda_test test_name target)
    add_test(NAME ${test_name} COMMAND ${target})
    set_tests_properties(${test_name} PROPERTIES
        ENVIRONMENT
            "OMNIVOICE_TEST_DEVICE=CUDA0;NVIDIA_TF32_OVERRIDE=0"
        LABELS "requires-assets;cuda"
        SKIP_RETURN_CODE 77
        TIMEOUT 120
    )
endfunction()

omnivoice_add_test(test_types omnivoice_types)
omnivoice_add_test(test_wav_writer omnivoice_wav_writer)
omnivoice_add_test(test_audio_io omnivoice_audio_io)
omnivoice_add_test(test_audio_postprocess omnivoice_audio_postprocess)
omnivoice_add_test(test_audio_resampler omnivoice_audio_resampler)
omnivoice_add_test(test_config omnivoice_config)
omnivoice_add_test(test_gguf_model omnivoice_gguf_model)
omnivoice_add_test(test_prompt_builder omnivoice_prompt_builder)
omnivoice_add_test(test_duration_estimator omnivoice_duration_estimator)
omnivoice_add_test(test_qwen3_block omnivoice_qwen3_block)
if(OMNIVOICE_ENABLE_CUDA)
    omnivoice_add_cuda_test(
        omnivoice_qwen3_block_cuda
        test_qwen3_block
    )
endif()

omnivoice_add_test(test_omnivoice_lm omnivoice_lm)
target_compile_definitions(test_omnivoice_lm PRIVATE
    OMNIVOICE_LM_GGUF="${OMNIVOICE_MODEL_DIR}/omnivoice-lm-f32.gguf"
)
set_tests_properties(omnivoice_lm PROPERTIES
    SKIP_RETURN_CODE 77
    TIMEOUT 120
)

omnivoice_add_test(test_mask_schedule omnivoice_mask_schedule)
omnivoice_add_test(test_token_selector omnivoice_token_selector)

omnivoice_add_test(test_merged_diffusion omnivoice_merged_diffusion)
target_compile_definitions(test_merged_diffusion PRIVATE
    OMNIVOICE_LM_GGUF="${OMNIVOICE_MODEL_DIR}/omnivoice-lm-f32.gguf"
)
set_tests_properties(omnivoice_merged_diffusion PROPERTIES
    SKIP_RETURN_CODE 77
    TIMEOUT 120
)
if(OMNIVOICE_ENABLE_CUDA)
    omnivoice_add_cuda_test(
        omnivoice_merged_diffusion_cuda
        test_merged_diffusion
    )
endif()

omnivoice_add_test(test_diffusion_decoder omnivoice_diffusion_decoder)
target_compile_definitions(test_diffusion_decoder PRIVATE
    OMNIVOICE_LM_GGUF="${OMNIVOICE_MODEL_DIR}/omnivoice-lm-f32.gguf"
)
set_tests_properties(omnivoice_diffusion_decoder PROPERTIES
    SKIP_RETURN_CODE 77
    TIMEOUT 120
)
if(OMNIVOICE_ENABLE_CUDA)
    omnivoice_add_cuda_test(
        omnivoice_diffusion_decoder_cuda
        test_diffusion_decoder
    )
endif()

omnivoice_add_test(test_higgs_decoder omnivoice_higgs_decoder)
target_compile_definitions(test_higgs_decoder PRIVATE
    OMNIVOICE_AUDIO_GGUF="${OMNIVOICE_MODEL_DIR}/omnivoice-tokenizer-f32.gguf"
)
set_tests_properties(omnivoice_higgs_decoder PROPERTIES
    SKIP_RETURN_CODE 77
    TIMEOUT 120
)
if(OMNIVOICE_ENABLE_CUDA)
    omnivoice_add_cuda_test(
        omnivoice_higgs_decoder_cuda
        test_higgs_decoder
    )
endif()

omnivoice_add_test(test_higgs_encoder omnivoice_higgs_encoder)
target_compile_definitions(test_higgs_encoder PRIVATE
    OMNIVOICE_AUDIO_GGUF="${OMNIVOICE_MODEL_DIR}/omnivoice-tokenizer-f32.gguf"
)
set_tests_properties(omnivoice_higgs_encoder PROPERTIES
    SKIP_RETURN_CODE 77
    TIMEOUT 180
)
if(OMNIVOICE_ENABLE_CUDA)
    omnivoice_add_cuda_test(
        omnivoice_higgs_encoder_cuda
        test_higgs_encoder
    )
    set_tests_properties(omnivoice_higgs_encoder_cuda PROPERTIES
        TIMEOUT 180
    )
endif()

omnivoice_add_test(test_text_tokenizer omnivoice_text_tokenizer)
omnivoice_add_test(test_text_chunker omnivoice_text_chunker)
omnivoice_add_test(
    test_voice_clone_prompt
    omnivoice_voice_clone_prompt
)
set_tests_properties(
    omnivoice_types
    omnivoice_wav_writer
    omnivoice_audio_io
    omnivoice_audio_resampler
    omnivoice_gguf_model
    omnivoice_voice_clone_prompt
    PROPERTIES
    LABELS "portable"
)

omnivoice_add_test(test_ggml_runtime omnivoice_ggml_runtime)
target_compile_definitions(test_ggml_runtime PRIVATE
    OMNIVOICE_LM_GGUF="${OMNIVOICE_MODEL_DIR}/omnivoice-lm-f32.gguf"
    OMNIVOICE_AUDIO_GGUF="${OMNIVOICE_MODEL_DIR}/omnivoice-tokenizer-f32.gguf"
)
set_tests_properties(omnivoice_ggml_runtime PROPERTIES
    SKIP_RETURN_CODE 77
    TIMEOUT 120
)
if(OMNIVOICE_ENABLE_CUDA)
    omnivoice_add_cuda_test(
        omnivoice_ggml_runtime_cuda
        test_ggml_runtime
    )
endif()

omnivoice_add_test(test_voice_clone omnivoice_voice_clone)
target_compile_definitions(test_voice_clone PRIVATE
    OMNIVOICE_LM_GGUF="${OMNIVOICE_MODEL_DIR}/omnivoice-lm-f32.gguf"
    OMNIVOICE_AUDIO_GGUF="${OMNIVOICE_MODEL_DIR}/omnivoice-tokenizer-f32.gguf"
)
set_tests_properties(omnivoice_voice_clone PROPERTIES
    SKIP_RETURN_CODE 77
    TIMEOUT 600
)

omnivoice_add_test(test_voice_clone_e2e omnivoice_voice_clone_e2e)
target_compile_definitions(test_voice_clone_e2e PRIVATE
    OMNIVOICE_LM_GGUF="${OMNIVOICE_MODEL_DIR}/omnivoice-lm-f32.gguf"
    OMNIVOICE_AUDIO_GGUF="${OMNIVOICE_MODEL_DIR}/omnivoice-tokenizer-f32.gguf"
)
set_tests_properties(omnivoice_voice_clone_e2e PROPERTIES
    SKIP_RETURN_CODE 77
    TIMEOUT 300
)

add_executable(
    test_voice_clone_extended
    test_voice_clone_extended.cpp
)
omnivoice_configure_target(test_voice_clone_extended)
target_link_libraries(
    test_voice_clone_extended
    PRIVATE omnivoice-core
)
target_compile_definitions(test_voice_clone_extended PRIVATE
    OMNIVOICE_TEST_DATA_DIR="${OMNIVOICE_TEST_DATA_DIR}"
    OMNIVOICE_LM_GGUF="${OMNIVOICE_MODEL_DIR}/omnivoice-lm-q8_0.gguf"
    OMNIVOICE_AUDIO_GGUF="${OMNIVOICE_MODEL_DIR}/omnivoice-tokenizer-f32.gguf"
)
if(OMNIVOICE_ENABLE_CUDA)
    add_test(
        NAME omnivoice_voice_clone_q8_cuda
        COMMAND test_voice_clone_extended
    )
    set_tests_properties(omnivoice_voice_clone_q8_cuda PROPERTIES
        ENVIRONMENT
            "OMNIVOICE_TEST_DEVICE=CUDA0;NVIDIA_TF32_OVERRIDE=0"
        LABELS "requires-assets;cuda"
        SKIP_RETURN_CODE 77
        TIMEOUT 600
    )
endif()

omnivoice_add_test(test_omnivoice_lm_prompt omnivoice_lm_prompt)
target_compile_definitions(test_omnivoice_lm_prompt PRIVATE
    OMNIVOICE_LM_GGUF="${OMNIVOICE_MODEL_DIR}/omnivoice-lm-f32.gguf"
)
set_tests_properties(omnivoice_lm_prompt PROPERTIES
    SKIP_RETURN_CODE 77
    TIMEOUT 600
)
if(OMNIVOICE_ENABLE_CUDA)
    add_test(
        NAME omnivoice_lm_prompt_cuda
        COMMAND test_omnivoice_lm_prompt
    )
    set_tests_properties(omnivoice_lm_prompt_cuda PROPERTIES
        ENVIRONMENT
            "OMNIVOICE_TEST_DEVICE=CUDA0;NVIDIA_TF32_OVERRIDE=0"
        LABELS "requires-assets;cuda"
        SKIP_RETURN_CODE 77
        TIMEOUT 600
    )
    add_test(
        NAME omnivoice_voice_clone_cuda
        COMMAND test_voice_clone
    )
    set_tests_properties(omnivoice_voice_clone_cuda PROPERTIES
        ENVIRONMENT
            "OMNIVOICE_TEST_DEVICE=CUDA0;NVIDIA_TF32_OVERRIDE=0"
        LABELS "requires-assets;cuda"
        SKIP_RETURN_CODE 77
        TIMEOUT 600
    )
    add_test(
        NAME omnivoice_voice_clone_e2e_cuda
        COMMAND test_voice_clone_e2e
    )
    set_tests_properties(omnivoice_voice_clone_e2e_cuda PROPERTIES
        ENVIRONMENT
            "OMNIVOICE_TEST_DEVICE=CUDA0;NVIDIA_TF32_OVERRIDE=0"
        LABELS "requires-assets;cuda"
        SKIP_RETURN_CODE 77
        TIMEOUT 300
    )
endif()

add_test(
    NAME omnivoice_cli_version
    COMMAND $<TARGET_FILE:omnivoice-cli> --version
)
add_test(
    NAME omnivoice_cli_list_devices
    COMMAND $<TARGET_FILE:omnivoice-cli> --list-devices
)
add_test(
    NAME omnivoice_cli_help
    COMMAND $<TARGET_FILE:omnivoice-cli> --help
)
add_test(
    NAME omnivoice_cli_unknown_option
    COMMAND $<TARGET_FILE:omnivoice-cli> --unknown-option
)
set_tests_properties(omnivoice_cli_unknown_option PROPERTIES WILL_FAIL TRUE)
add_test(
    NAME omnivoice_cli_clone_requires_args
    COMMAND $<TARGET_FILE:omnivoice-cli> --clone
)
set_tests_properties(
    omnivoice_cli_clone_requires_args
    PROPERTIES WILL_FAIL TRUE
)
add_test(
    NAME omnivoice_cli_prompt_requires_args
    COMMAND $<TARGET_FILE:omnivoice-cli> --create-prompt
)
set_tests_properties(
    omnivoice_cli_prompt_requires_args
    PROPERTIES WILL_FAIL TRUE
)
set_tests_properties(
    omnivoice_cli_version
    omnivoice_cli_list_devices
    omnivoice_cli_help
    omnivoice_cli_unknown_option
    omnivoice_cli_clone_requires_args
    omnivoice_cli_prompt_requires_args
    PROPERTIES
    LABELS "portable;cli"
)

omnivoice_add_test(test_cli_tone omnivoice_cli_tone)
target_compile_definitions(test_cli_tone PRIVATE
    OMNIVOICE_CLI_PATH="$<TARGET_FILE:omnivoice-cli>"
)
add_dependencies(test_cli_tone omnivoice-cli)
set_tests_properties(omnivoice_cli_tone PROPERTIES
    LABELS "portable;cli"
    TIMEOUT 60
)
