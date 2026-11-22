fn main() {
    println!("cargo:rerun-if-changed=third_party/soxr");
    let msvc = std::env::var("CARGO_CFG_TARGET_ENV").ok().as_deref() == Some("msvc");
    let linux = std::env::var("CARGO_CFG_TARGET_OS").ok().as_deref() == Some("linux");
    let gnu = std::env::var("CARGO_CFG_TARGET_ENV").ok().as_deref() == Some("gnu");
    let x86_64 = std::env::var("CARGO_CFG_TARGET_ARCH").ok().as_deref() == Some("x86_64");
    // glibc libmvec + AVX-512 C wrapper: only on x86_64 Linux with glibc.
    // The OS+ENV check is required so Windows-gnu and macOS never spawn `cc
    // -mavx512f` or link `-lmvec`. The arch check avoids the same failure on
    // aarch64-linux-gnu, which would otherwise match linux+gnu.
    let use_libmvec = linux && gnu && x86_64;

    let opt = if msvc { "/O2" } else { "-O2" };
    let dst = cmake::Config::new("third_party/soxr")
        .define("BUILD_SHARED_LIBS", "OFF")
        .define("BUILD_TESTS", "OFF")
        .define("BUILD_EXAMPLES", "OFF")
        .define("WITH_OPENMP", "OFF")
        .define("WITH_LSR_BINDINGS", "OFF")
        .define("WITH_DEV_TRACE", "OFF")
        .define("CMAKE_POSITION_INDEPENDENT_CODE", "ON")
        .cflag(opt)
        .profile("Release")
        .build();
    let lib = dst.join("lib");
    println!("cargo:rustc-link-search=native={}", lib.display());
    println!("cargo:rustc-link-lib=static=soxr");
    if !msvc {
        println!("cargo:rustc-link-lib=m");
    }

    if !use_libmvec {
        return;
    }

    // glibc libmvec wrappers (AVX-512 16-wide expf/erff) for encoder kernels.
    println!("cargo:rustc-link-lib=mvec");
    println!("cargo:rerun-if-changed=src/libmvec_wrap.c");
    let cc = std::env::var("CC").unwrap_or_else(|_| "cc".into());
    let out = std::env::var("OUT_DIR").expect("OUT_DIR");
    let obj = std::path::Path::new(&out).join("libmvec_wrap.o");
    let src = std::path::Path::new("src/libmvec_wrap.c");
    let status = std::process::Command::new(&cc)
        .args(["-O3", "-mavx512f", "-c"])
        .arg(src)
        .arg("-o")
        .arg(&obj)
        .status()
        .expect("spawn cc for libmvec_wrap");
    if !status.success() {
        panic!("failed to compile src/libmvec_wrap.c with {cc}");
    }
    let lib = std::path::Path::new(&out).join("libmvec_wrap.a");
    let ar = std::env::var("AR").unwrap_or_else(|_| "ar".into());
    let status = std::process::Command::new(&ar)
        .args(["crus"])
        .arg(&lib)
        .arg(&obj)
        .status()
        .expect("spawn ar");
    if !status.success() {
        panic!("failed to archive libmvec_wrap.o");
    }
    println!("cargo:rustc-link-search=native={out}");
    println!("cargo:rustc-link-lib=static=mvec_wrap");
}
