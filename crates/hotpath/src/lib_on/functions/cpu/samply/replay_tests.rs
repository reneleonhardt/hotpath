use std::collections::HashMap;
use std::ffi::OsString;
use std::fmt::Write as _;
use std::fs;
use std::process::Command;
use std::sync::LazyLock;

use object::{Object, ObjectSymbol};

use crate::functions::cpu::samply::pick_image_base;
use crate::functions::cpu::{build_cpu_report_from_path, CPU_INCLUSIVE};
use crate::json::CpuSampleUnit;

static RUSTC: LazyLock<OsString> =
    LazyLock::new(|| std::env::var_os("RUSTC").unwrap_or_else(|| "rustc".into()));

#[test]
fn replays_shared_library_attribution() {
    const ENTRY: &str = "hotpath_cpu_fixture::entry";
    const LEAF: &str = "hotpath_cpu_fixture::leaf";
    let dir = std::env::temp_dir().join(format!("hotpath-cpu-replay-{}", std::process::id()));
    fs::create_dir_all(&dir).unwrap();
    let source = dir.join("fixture.rs");
    let library = dir.join(format!("replay{}", std::env::consts::DLL_SUFFIX));
    fs::write(
        &source,
        r#"
#[inline(always)]
fn leaf(value: u64) -> u64 {
    std::hint::black_box(value).wrapping_mul(17)
}

#[inline(never)]
fn entry(value: u64) -> u64 {
    leaf(value)
}

#[no_mangle]
pub extern "C" fn replay_call(value: u64) -> u64 {
    entry(value)
}
"#,
    )
    .unwrap();
    let mut rustc = Command::new(&*RUSTC);
    rustc.args([
        "--edition=2021",
        "--crate-name",
        "hotpath_cpu_fixture",
        "--crate-type=cdylib",
        "-O",
        "-g",
    ]);
    if cfg!(target_os = "macos") {
        rustc.arg("-Csplit-debuginfo=packed");
    }
    let output = rustc.arg(&source).arg("-o").arg(&library).output().unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );

    let bytes = fs::read(&library).unwrap();
    let object = object::File::parse(bytes.as_slice()).unwrap();
    let entry = object
        .symbols()
        .find(|symbol| {
            symbol
                .name()
                .ok()
                .and_then(|name| rustc_demangle::try_demangle(name).ok())
                .is_some_and(|name| format!("{name:#}") == ENTRY)
        })
        .unwrap();
    let pc = entry.address();
    #[cfg(not(feature = "hotpath-cpu-inline"))]
    let expected = ENTRY;
    #[cfg(feature = "hotpath-cpu-inline")]
    let (pc, expected) = {
        let debug = if cfg!(target_os = "macos") {
            dir.join("replay.dylib.dSYM/Contents/Resources/DWARF/replay.dylib")
        } else {
            library.clone()
        };
        let loader = addr2line::Loader::new(debug).unwrap();
        let pc = (pc..pc + entry.size().max(128))
            .find(|&address| {
                let mut frames = loader.find_frames(address).unwrap();
                while let Some(frame) = frames.next().unwrap() {
                    if frame.function.is_some_and(|name| {
                        name.raw_name().is_ok_and(|raw| {
                            format!("{:#}", rustc_demangle::demangle(&raw)) == LEAF
                        })
                    }) {
                        return true;
                    }
                }
                false
            })
            .expect("fixture must have a real DWARF inline frame");
        (pc, LEAF)
    };
    let code_id = object.mach_uuid().unwrap().map(|uuid| {
        let mut hex = String::new();
        for byte in uuid {
            write!(hex, "{byte:02x}").unwrap();
        }
        hex
    });

    let pid = std::process::id();
    let thread = serde_json::json!({
        "pid": format!("{pid}.1"),
        "samples": {"stack": [0, 0, 0, 0], "threadCPUDelta": [2, 0, 5, 4]},
        "stackTable": {"prefix": [null], "frame": [0]},
        "frameTable": {"address": [pc - pick_image_base(&object)], "func": [0]},
        "funcTable": {"resource": [0]},
        "resourceTable": {"lib": [0]}
    });
    let mut earlier = thread.clone();
    earlier["pid"] = serde_json::json!(pid);
    earlier["samples"]["threadCPUDelta"] = serde_json::Value::Null;
    let mut foreign = thread.clone();
    foreign["pid"] = serde_json::json!(pid + 1);
    let path = dir.join("profile.json");
    let profile = serde_json::json!({
        "meta": {"sampleUnits": {"threadCPUDelta": "µs"}},
        "libs": [{"path": library, "codeId": code_id}],
        "threads": [earlier, foreign, thread],
    });
    fs::write(&path, profile.to_string()).unwrap();
    let expected_rows = if cfg!(feature = "hotpath-cpu-inline") && *CPU_INCLUSIVE {
        &[(ENTRY, 11), (LEAF, 11)][..]
    } else {
        &[(expected, 11)][..]
    };
    for _ in 0..2 {
        let names = HashMap::from([(ENTRY, 1), (LEAF, 2)]);
        let report = build_cpu_report_from_path("replay", &path, Some(names)).unwrap();
        assert_eq!(report.sample_unit, CpuSampleUnit::Microseconds);
        assert_eq!((report.total_samples, report.attributed_samples), (11, 11));
        let rows: Vec<_> = report
            .stats
            .iter()
            .map(|row| (row.name, row.samples))
            .collect();
        assert_eq!(rows, expected_rows);
    }
    fs::remove_dir_all(dir).unwrap();
}
