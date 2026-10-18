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
        .compile_protos(&["proto/snoozeml/v1/inference.proto"], &["proto"])?;

    println!("cargo:rerun-if-changed=proto/snoozeml/v1/inference.proto");
    Ok(())
}
