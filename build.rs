//! Build script for SnoozeML.
//!
//! Compiles Protocol Buffer schemas using `tonic-build` and `protoc-bin-vendored`,
//! configures Linux RPATH and `--as-needed` linker flags, and provides a lazy
//! `dlopen` shim for `libcuda.so.1` on Linux CUDA builds so the compiled binary
//! never carries a hard `DT_NEEDED` link-time dependency on `libcuda.so.1`.

use std::path::{Path, PathBuf};

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

    // Embed relative RPATHs (DT_RPATH via --disable-new-dtags) and --as-needed on Linux
    // so the binary and dlopen'd execution providers automatically discover bundled ONNX
    // Runtime libraries and host CUDA installations without requiring manual LD_LIBRARY_PATH
    // exports or binding unused shared libraries into DT_NEEDED.
    let target_os = std::env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    let target_arch = std::env::var("CARGO_CFG_TARGET_ARCH").unwrap_or_default();
    if target_os == "linux" {
        println!("cargo:rustc-link-arg=-Wl,--as-needed");
        println!(
            "cargo:rustc-link-arg=-Wl,--disable-new-dtags,-rpath,$ORIGIN/lib:$ORIGIN/../lib:$ORIGIN/../lib/snooze:$ORIGIN:/usr/local/cuda/lib64:/usr/local/cuda-13.0/lib64"
        );

        if std::env::var("CARGO_FEATURE_CANDLE_CUDA").is_ok() && target_arch == "x86_64" {
            build_linux_cuda_driver_dlopen_shim()?;
        }
    }

    Ok(())
}

/// Generates and links a static lazy-`dlopen` shim for `libcuda.so.1` on Linux x86_64.
///
/// Because `candle-core` enables `cudarc/dynamic-linking` (which emits `-lcuda` against
/// `stubs/libcuda.so`), providing strong symbol definitions that lazily `dlopen("libcuda.so.1")`
/// upon `cuInit` allows `-Wl,--as-needed` to strip `libcuda.so.1` from the binary's `DT_NEEDED`
/// table while preserving full GPU execution when an NVIDIA driver is present on the host.
fn build_linux_cuda_driver_dlopen_shim() -> Result<(), Box<dyn std::error::Error>> {
    let mut candidate_stubs = Vec::new();
    for env_key in [
        "CUDA_HOME",
        "CUDA_PATH",
        "CUDA_ROOT",
        "CUDA_TOOLKIT_ROOT_DIR",
    ] {
        if let Ok(root) = std::env::var(env_key) {
            candidate_stubs.push(PathBuf::from(&root).join("lib64/stubs/libcuda.so"));
            candidate_stubs.push(PathBuf::from(&root).join("lib/stubs/libcuda.so"));
        }
    }
    candidate_stubs.push(PathBuf::from("/usr/local/cuda/lib64/stubs/libcuda.so"));
    candidate_stubs.push(PathBuf::from("/usr/local/cuda-13.0/lib64/stubs/libcuda.so"));
    candidate_stubs.push(PathBuf::from("/usr/lib/x86_64-linux-gnu/stubs/libcuda.so"));

    let Some(stub_path) = candidate_stubs.into_iter().find(|p| p.exists()) else {
        return Ok(());
    };

    let nm_out = std::process::Command::new("nm")
        .args(["-D", "--defined-only", "-f", "posix"])
        .arg(&stub_path)
        .output()?;

    if !nm_out.status.success() {
        return Ok(());
    }

    let stdout = String::from_utf8_lossy(&nm_out.stdout);
    let mut symbols = Vec::new();
    for line in stdout.lines() {
        let mut parts = line.split_whitespace();
        let Some(raw_name) = parts.next() else {
            continue;
        };
        let Some(sym_type) = parts.next() else {
            continue;
        };
        if sym_type != "T" && sym_type != "W" {
            continue;
        }
        let sym_name = raw_name.split('@').next().unwrap_or(raw_name);
        if sym_name.starts_with("cu")
            && sym_name != "cuInit"
            && sym_name != "cuGetErrorName"
            && sym_name != "cuGetErrorString"
        {
            symbols.push(sym_name.to_string());
        }
    }
    symbols.sort();
    symbols.dedup();

    if symbols.is_empty() {
        return Ok(());
    }

    let out_dir = PathBuf::from(std::env::var("OUT_DIR")?);
    let c_path = out_dir.join("snooze_cuda_driver_shim.c");
    let s_path = out_dir.join("snooze_cuda_driver_trampolines.S");
    let c_obj = out_dir.join("snooze_cuda_driver_shim.o");
    let s_obj = out_dir.join("snooze_cuda_driver_trampolines.o");
    let lib_path = out_dir.join("libsnooze_cuda_driver_shim.a");

    write_shim_sources(&c_path, &s_path, &symbols)?;

    let cc_status = std::process::Command::new("cc")
        .args(["-O2", "-fPIC", "-c"])
        .arg(&c_path)
        .arg("-o")
        .arg(&c_obj)
        .status()?;
    if !cc_status.success() {
        return Err("Failed to compile snooze_cuda_driver_shim.c".into());
    }

    let as_status = std::process::Command::new("cc")
        .args(["-O2", "-fPIC", "-c"])
        .arg(&s_path)
        .arg("-o")
        .arg(&s_obj)
        .status()?;
    if !as_status.success() {
        return Err("Failed to compile snooze_cuda_driver_trampolines.S".into());
    }

    let _ = std::fs::remove_file(&lib_path);
    let ar_status = std::process::Command::new("ar")
        .args(["rcs"])
        .arg(&lib_path)
        .arg(&c_obj)
        .arg(&s_obj)
        .status()?;
    if !ar_status.success() {
        return Err("Failed to archive libsnooze_cuda_driver_shim.a".into());
    }

    println!("cargo:rustc-link-search=native={}", out_dir.display());
    println!("cargo:rustc-link-lib=static:+whole-archive=snooze_cuda_driver_shim");
    println!("cargo:rustc-link-lib=dylib=dl");

    Ok(())
}

fn write_shim_sources(
    c_path: &Path,
    s_path: &Path,
    symbols: &[String],
) -> Result<(), Box<dyn std::error::Error>> {
    use std::fmt::Write as _;

    let mut c_code = String::with_capacity(64 * 1024);
    writeln!(c_code, "#define _GNU_SOURCE")?;
    writeln!(c_code, "#include <dlfcn.h>")?;
    writeln!(c_code, "#include <pthread.h>")?;
    writeln!(c_code, "#include <stddef.h>")?;
    writeln!(c_code)?;
    writeln!(c_code, "#define CUDA_SUCCESS 0")?;
    writeln!(c_code, "#define CUDA_ERROR_STUB_LIBRARY 34")?;
    writeln!(c_code, "#define CUDA_ERROR_NOT_INITIALIZED 3")?;
    writeln!(c_code)?;
    writeln!(
        c_code,
        "static pthread_once_t g_snooze_cuda_once = PTHREAD_ONCE_INIT;"
    )?;
    writeln!(c_code, "static void *g_snooze_libcuda = NULL;")?;
    writeln!(c_code, "static int (*g_real_cuInit)(unsigned int) = NULL;")?;
    writeln!(
        c_code,
        "static int (*g_real_cuGetErrorName)(int, const char **) = NULL;"
    )?;
    writeln!(
        c_code,
        "static int (*g_real_cuGetErrorString)(int, const char **) = NULL;"
    )?;
    writeln!(c_code)?;

    for sym in symbols {
        writeln!(c_code, "void *__snooze_ptr_{sym} = NULL;")?;
    }

    writeln!(c_code)?;
    writeln!(c_code, "static void snooze_load_libcuda(void) {{")?;
    writeln!(
        c_code,
        "    const char *candidates[] = {{ \"libcuda.so.1\", \"libcuda.so\", NULL }};"
    )?;
    writeln!(c_code, "    for (int i = 0; candidates[i] != NULL; ++i) {{")?;
    writeln!(
        c_code,
        "        g_snooze_libcuda = dlopen(candidates[i], RTLD_LAZY | RTLD_LOCAL);"
    )?;
    writeln!(c_code, "        if (g_snooze_libcuda != NULL) break;")?;
    writeln!(c_code, "    }}")?;
    writeln!(c_code, "    if (g_snooze_libcuda == NULL) return;")?;
    writeln!(
        c_code,
        "    g_real_cuInit = (int (*)(unsigned int))dlsym(g_snooze_libcuda, \"cuInit\");"
    )?;
    writeln!(
        c_code,
        "    g_real_cuGetErrorName = (int (*)(int, const char **))dlsym(g_snooze_libcuda, \"cuGetErrorName\");"
    )?;
    writeln!(
        c_code,
        "    g_real_cuGetErrorString = (int (*)(int, const char **))dlsym(g_snooze_libcuda, \"cuGetErrorString\");"
    )?;
    for sym in symbols {
        writeln!(
            c_code,
            "    __snooze_ptr_{sym} = dlsym(g_snooze_libcuda, \"{sym}\");"
        )?;
    }
    writeln!(c_code, "}}")?;
    writeln!(c_code)?;
    writeln!(c_code, "int cuInit(unsigned int flags) {{")?;
    writeln!(
        c_code,
        "    pthread_once(&g_snooze_cuda_once, snooze_load_libcuda);"
    )?;
    writeln!(c_code, "    if (g_real_cuInit == NULL) {{")?;
    writeln!(c_code, "        return CUDA_ERROR_STUB_LIBRARY;")?;
    writeln!(c_code, "    }}")?;
    writeln!(c_code, "    return g_real_cuInit(flags);")?;
    writeln!(c_code, "}}")?;
    writeln!(c_code)?;
    writeln!(
        c_code,
        "int cuGetErrorName(int error, const char **pStr) {{"
    )?;
    writeln!(
        c_code,
        "    pthread_once(&g_snooze_cuda_once, snooze_load_libcuda);"
    )?;
    writeln!(c_code, "    if (g_real_cuGetErrorName != NULL) {{")?;
    writeln!(c_code, "        return g_real_cuGetErrorName(error, pStr);")?;
    writeln!(c_code, "    }}")?;
    writeln!(c_code, "    if (pStr != NULL) {{")?;
    writeln!(
        c_code,
        "        *pStr = (error == CUDA_ERROR_STUB_LIBRARY) ? \"CUDA_ERROR_STUB_LIBRARY\" : \"CUDA_ERROR_NOT_INITIALIZED\";"
    )?;
    writeln!(c_code, "    }}")?;
    writeln!(c_code, "    return CUDA_SUCCESS;")?;
    writeln!(c_code, "}}")?;
    writeln!(c_code)?;
    writeln!(
        c_code,
        "int cuGetErrorString(int error, const char **pStr) {{"
    )?;
    writeln!(
        c_code,
        "    pthread_once(&g_snooze_cuda_once, snooze_load_libcuda);"
    )?;
    writeln!(c_code, "    if (g_real_cuGetErrorString != NULL) {{")?;
    writeln!(
        c_code,
        "        return g_real_cuGetErrorString(error, pStr);"
    )?;
    writeln!(c_code, "    }}")?;
    writeln!(c_code, "    if (pStr != NULL) {{")?;
    writeln!(
        c_code,
        "        *pStr = (error == CUDA_ERROR_STUB_LIBRARY) ? \"NVIDIA driver library (libcuda.so.1) not found on host\" : \"CUDA driver not initialized\";"
    )?;
    writeln!(c_code, "    }}")?;
    writeln!(c_code, "    return CUDA_SUCCESS;")?;
    writeln!(c_code, "}}")?;

    std::fs::write(c_path, c_code)?;

    let mut s_code = String::with_capacity(64 * 1024);
    writeln!(s_code, ".text")?;
    writeln!(s_code, ".p2align 4")?;
    writeln!(s_code, ".L_snooze_cuda_uninit:")?;
    writeln!(s_code, "    movl $34, %eax")?;
    writeln!(s_code, "    ret")?;
    writeln!(s_code)?;

    for sym in symbols {
        writeln!(s_code, ".globl {sym}")?;
        writeln!(s_code, ".type {sym}, @function")?;
        writeln!(s_code, "{sym}:")?;
        writeln!(s_code, "    cmpq $0, __snooze_ptr_{sym}(%rip)")?;
        writeln!(s_code, "    je .L_snooze_cuda_uninit")?;
        writeln!(s_code, "    jmp *__snooze_ptr_{sym}(%rip)")?;
        writeln!(s_code, ".size {sym}, .-{sym}")?;
        writeln!(s_code)?;
    }

    writeln!(s_code, ".section .note.GNU-stack,\"\",@progbits")?;
    std::fs::write(s_path, s_code)?;

    Ok(())
}
