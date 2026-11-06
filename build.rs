//! Build script for SnoozeML.
//!
//! Compiles Protocol Buffer schemas using `tonic-build` and `protoc-bin-vendored`,
//! and configures Linux / macOS RPATH linker flags for dynamic runtime discovery of ONNX Runtime.

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

    let target_os = std::env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    match target_os.as_str() {
        "linux" => {
            println!(
                "cargo:rustc-link-arg=-Wl,--disable-new-dtags,-rpath,$ORIGIN/lib:$ORIGIN/../lib:$ORIGIN/../lib/snooze:$ORIGIN:/usr/local/cuda/lib64:/usr/local/cuda-13.0/lib64:/usr/local/cuda-12.0/lib64"
            );
        }
        "macos" => {
            println!("cargo:rustc-link-arg=-Wl,-rpath,@executable_path/lib");
            println!("cargo:rustc-link-arg=-Wl,-rpath,@executable_path/../lib");
            println!("cargo:rustc-link-arg=-Wl,-rpath,@executable_path");
        }
        _ => {}
    }

    Ok(())
}
