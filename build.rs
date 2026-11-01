//! Build script for SnoozeML.
//!
//! Compiles Protocol Buffer schemas using `tonic-build` and `protoc-bin-vendored`.

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let protoc_path = protoc_bin_vendored::protoc_bin_path()
        .map_err(|e| format!("Failed to locate vendored protoc binary: {e}"))?;
    std::env::set_var("PROTOC", protoc_path);

    tonic_build::configure()
        .build_server(true)
        .build_client(true)
        .compile_protos(
            &[
                "proto/snoozeml/v1/inference.proto",
                "proto/third_party/grpc_service.proto",
            ],
            &["proto", "proto/third_party"],
        )?;

    println!("cargo:rerun-if-changed=proto/snoozeml/v1/inference.proto");
    println!("cargo:rerun-if-changed=proto/third_party/grpc_service.proto");
    println!("cargo:rerun-if-changed=proto/third_party/model_config.proto");
    println!("cargo:rerun-if-changed=proto/third_party/health.proto");

    // Embed relative RPATHs (DT_RPATH via --disable-new-dtags) on Linux so the binary and
    // dlopen'd execution providers automatically discover bundled ONNX Runtime libraries and
    // host CUDA installations without requiring manual LD_LIBRARY_PATH exports.
    let target_os = std::env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    if target_os == "linux" {
        println!(
            "cargo:rustc-link-arg=-Wl,--disable-new-dtags,-rpath,$ORIGIN/lib:$ORIGIN/../lib:$ORIGIN/../lib/snooze:$ORIGIN:/usr/local/cuda/lib64:/usr/local/cuda-13.0/lib64"
        );
    }

    Ok(())
}
