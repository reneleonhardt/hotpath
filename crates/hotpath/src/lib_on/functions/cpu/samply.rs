use std::borrow::Cow;
use std::collections::{HashMap, HashSet};
use std::fmt::Write as _;
use std::fs::File;
use std::io::Read;
use std::path::Path;
use std::sync::{Arc, Mutex};

use crate::dev_logging::{info, warn};
use object::{Object, ObjectSegment, ObjectSymbol, SymbolKind};

use crate::lib_on::functions::cpu::json::{Lib, Profile};
use crate::lib_on::functions::cpu::{CpuFunctionStats, CpuReport, CPU_INCLUSIVE};

#[cfg_attr(feature = "hotpath-meta", hotpath_meta::measure)]
pub(crate) fn build_cpu_report_from_path(
    caller_name: &'static str,
    path: &Path,
) -> Option<CpuReport> {
    info!("cpu report: loading samply profile from {}", path.display());

    let profile = match load_profile(path) {
        Ok(p) => p,
        Err(e) => {
            warn!(
                "cpu report: failed to load samply profile {}: {e}",
                path.display()
            );
            return None;
        }
    };

    let display_to_id = match crate::lib_on::functions::get_instrumented_names_and_ids() {
        Some(display_to_id) => display_to_id,
        None => {
            warn!("cpu report: instrumented function registry unavailable; skipping CPU report");
            return None;
        }
    };

    if display_to_id.is_empty() {
        warn!("cpu report: no instrumented functions registered; CPU report empty");
        return None;
    }

    let label_to_symbol = crate::lib_on::functions::get_cpu_label_aliases();
    let mut match_to_display: HashMap<&'static str, &'static str> =
        HashMap::with_capacity(display_to_id.len());
    for &display in display_to_id.keys() {
        let match_key = label_to_symbol.get(display).copied().unwrap_or(display);
        match_to_display.insert(match_key, display);
    }

    let primary_lib_name = std::env::current_exe()
        .ok()
        .and_then(|p| p.file_name().map(|n| n.to_os_string()));
    if primary_lib_name.is_none() {
        warn!("cpu report: could not resolve current_exe; CPU report will be empty");
    }

    let primary: Option<(usize, Arc<LibSymbolIndex>)> =
        profile.libs.iter().enumerate().find_map(|(i, lib)| {
            let is_primary = primary_lib_name.as_deref().is_some_and(|name| {
                lib.path
                    .iter()
                    .chain(&lib.debug_path)
                    .any(|path| Path::new(path).file_name() == Some(name))
            });
            if !is_primary {
                return None;
            }
            let idx = build_lib_index(lib, &match_to_display)?;
            Some((i, idx))
        });
    let total_matches: usize = primary.as_ref().map(|(_, i)| i.ranges.len()).unwrap_or(0);
    if total_matches == 0 {
        warn!(
            "cpu report: no instrumented symbols found in any sampled library - ensure the binary was built with debug symbols and not stripped"
        );
    }

    let sample_unit = profile.sample_unit();
    let mut sample_counts: HashMap<&'static str, u64> = HashMap::new();
    let mut total_samples: u64 = 0;
    let mut attributed_samples: u64 = 0;
    let inclusive = *CPU_INCLUSIVE;
    let mut matched = HashSet::new();
    let mut weights_by_root = Vec::new();

    for thread in &profile.threads {
        let prefix = &thread.stack_table.prefix;
        let stack_frame = &thread.stack_table.frame;
        let frame_addr = &thread.frame_table.address;
        let frame_func = &thread.frame_table.func;
        let func_resource = &thread.func_table.resource;
        let resource_lib = &thread.resource_table.lib;

        let attribute = |root, weight| {
            let mut credited = false;
            let mut cur = root;
            while let Some(s) = cur {
                let frame_idx = match stack_frame.get(s) {
                    Some(f) => *f,
                    None => break,
                };
                let address = frame_addr.get(frame_idx).copied().unwrap_or(-1);
                let func_idx = frame_func.get(frame_idx).copied();
                let lib_opt = func_idx
                    .and_then(|fi| func_resource.get(fi).copied())
                    .filter(|r| *r >= 0)
                    .and_then(|r| resource_lib.get(r as usize).copied().flatten())
                    .filter(|l| *l >= 0)
                    .map(|l| l as usize);

                if address >= 0 {
                    if let (Some(lib_idx), Some((primary_idx, primary_lib))) =
                        (lib_opt, primary.as_ref())
                    {
                        if lib_idx == *primary_idx {
                            if let Some(sym) = primary_lib.lookup(address as u64) {
                                if inclusive {
                                    matched.insert(sym);
                                } else {
                                    *sample_counts.entry(sym).or_insert(0) += weight;
                                    credited = true;
                                    break;
                                }
                            }
                        }
                    }
                }

                // Samply interns parents before children. Stop on a malformed link.
                cur = prefix
                    .get(s)
                    .copied()
                    .flatten()
                    .filter(|&parent| parent < s);
            }

            if inclusive && !matched.is_empty() {
                for sym in matched.drain() {
                    *sample_counts.entry(sym).or_insert(0) += weight;
                }
                attributed_samples += weight;
            } else if credited {
                attributed_samples += weight;
            }
        };
        total_samples += visit_sample_stacks(
            &thread.samples.stack,
            thread.samples.weights(sample_unit),
            stack_frame.len(),
            &mut weights_by_root,
            attribute,
        );
    }

    let mut stats: Vec<CpuFunctionStats> = sample_counts
        .into_iter()
        .filter_map(|(name, samples)| {
            display_to_id.get(name).map(|id| CpuFunctionStats {
                name,
                id: *id,
                samples,
            })
        })
        .collect();

    stats.sort_by(|a, b| b.samples.cmp(&a.samples).then_with(|| a.name.cmp(b.name)));

    info!(
        "cpu report: total_samples={total_samples} attributed_samples={attributed_samples} matched_symbols={total_matches} stats_rows={}",
        stats.len()
    );

    if attributed_samples == 0 {
        warn!(
            "cpu report: no samples were attributed to instrumented functions; total_samples={total_samples} matched_symbols={total_matches}"
        );
    } else if stats.is_empty() {
        warn!(
            "cpu report: CPU profile parsed but produced no stats rows; total_samples={total_samples} attributed_samples={attributed_samples} matched_symbols={total_matches}"
        );
    }
    Some(CpuReport {
        total_samples,
        attributed_samples,
        sample_unit,
        caller_name,
        stats,
        profile_path: path.display().to_string(),
    })
}

#[cfg_attr(feature = "hotpath-meta", hotpath_meta::measure)]
fn load_profile(path: &std::path::Path) -> Result<Profile, Box<dyn std::error::Error>> {
    let mut file = File::open(path)?;
    let mut buf = Vec::new();
    file.read_to_end(&mut buf)?;

    let bytes: Vec<u8> = if buf.starts_with(&[0x1f, 0x8b]) {
        let mut decoder = flate2::read::GzDecoder::new(&buf[..]);
        let mut decoded = Vec::new();
        decoder.read_to_end(&mut decoded)?;
        decoded
    } else {
        buf
    };

    Ok(serde_json::from_slice::<Profile>(&bytes)?)
}

#[derive(Default)]
struct LibSymbolIndex {
    ranges: Vec<(u64, u64, &'static str)>,
}

#[cfg_attr(feature = "hotpath-meta", hotpath_meta::measure_all)]
impl LibSymbolIndex {
    fn lookup(&self, addr: u64) -> Option<&'static str> {
        let idx = self.ranges.partition_point(|(start, _, _)| *start <= addr);
        let (_, end, sym) = self.ranges[idx.checked_sub(1)?];
        (addr < end).then_some(sym)
    }
}

type FileStamp = (u64, u64, u64, i64, i64);

#[cfg(unix)]
fn file_stamp(file: &File) -> Option<FileStamp> {
    use std::os::unix::fs::MetadataExt;

    let metadata = file.metadata().ok()?;
    Some((
        metadata.dev(),
        metadata.ino(),
        metadata.len(),
        metadata.ctime(),
        metadata.ctime_nsec(),
    ))
}

#[cfg(not(unix))]
fn file_stamp(_: &File) -> Option<FileStamp> {
    None
}

struct CachedSymbolIndex {
    stamp: FileStamp,
    labels: HashMap<&'static str, &'static str>,
    index: Arc<LibSymbolIndex>,
}

#[derive(Default)]
struct SymbolCache(Mutex<Option<CachedSymbolIndex>>);

impl SymbolCache {
    fn index(
        &self,
        mut file: File,
        labels: &HashMap<&'static str, &'static str>,
    ) -> Option<Arc<LibSymbolIndex>> {
        let stamp = file_stamp(&file);
        if let Ok(cache) = self.0.lock() {
            if let Some(cached) = cache.as_ref() {
                if Some(cached.stamp) == stamp && &cached.labels == labels {
                    return Some(Arc::clone(&cached.index));
                }
            }
        }

        let mut bytes = Vec::new();
        file.read_to_end(&mut bytes).ok()?;
        let parsed = object::File::parse(&*bytes).ok()?;
        let index = Arc::new(build_symbol_index(&parsed, labels));
        if let Some(stamp) = stamp.filter(|&stamp| Some(stamp) == file_stamp(&file)) {
            if let Ok(mut cache) = self.0.lock() {
                *cache = Some(CachedSymbolIndex {
                    stamp,
                    labels: labels.clone(),
                    index: Arc::clone(&index),
                });
            }
        }
        Some(index)
    }
}

#[cfg_attr(feature = "hotpath-meta", hotpath_meta::measure)]
fn build_lib_index(
    lib: &Lib,
    match_to_display: &HashMap<&'static str, &'static str>,
) -> Option<Arc<LibSymbolIndex>> {
    static CACHE: SymbolCache = SymbolCache(Mutex::new(None));

    let debug_path = lib.debug_path.as_deref().filter(|p| !p.is_empty());
    let binary_path = lib.path.as_deref().filter(|&path| Some(path) != debug_path);
    debug_path.into_iter().chain(binary_path).find_map(|path| {
        let file = File::open(path).ok()?;
        CACHE.index(file, match_to_display)
    })
}

fn build_symbol_index(
    parsed: &object::File<'_>,
    match_to_display: &HashMap<&'static str, &'static str>,
) -> LibSymbolIndex {
    let base = pick_image_base(parsed);

    let mut all_starts: Vec<u64> = Vec::new();
    let mut matched_pending: Vec<(u64, u64, &'static str)> = Vec::new();
    let mut demangled = String::new();
    for sym in parsed.symbols() {
        if !matches!(sym.kind(), SymbolKind::Text) {
            continue;
        }
        let rel = sym.address().saturating_sub(base);
        all_starts.push(rel);

        let raw_name = match sym.name() {
            Ok(n) if !n.is_empty() => n,
            _ => continue,
        };
        demangled.clear();
        write!(demangled, "{:#}", rustc_demangle::demangle(raw_name)).unwrap();
        let normalized = strip_hash_suffix(&demangled);
        let normalized = strip_inherent_impl_brackets(normalized);
        if let Some(display) = match_eligible_symbol(&normalized, match_to_display) {
            matched_pending.push((rel, sym.size(), display));
        }
    }
    all_starts.sort_unstable();
    all_starts.dedup();

    let mut ranges: Vec<(u64, u64, &'static str)> = matched_pending
        .into_iter()
        .map(|(rel, declared, name)| {
            let size = if declared > 0 {
                declared
            } else {
                let idx = all_starts.partition_point(|s| *s <= rel);
                all_starts
                    .get(idx)
                    .map(|next| next.saturating_sub(rel))
                    .filter(|s| *s > 0)
                    .unwrap_or(1)
            };
            (rel, rel.saturating_add(size), name)
        })
        .collect();

    ranges.sort_by_key(|(start, _, _)| *start);

    LibSymbolIndex { ranges }
}

fn pick_image_base<'a>(file: &object::File<'a, &'a [u8]>) -> u64 {
    let rel = file.relative_address_base();
    if rel != 0 {
        return rel;
    }
    file.segments()
        .filter_map(|seg| {
            let name = seg.name().ok().flatten()?;
            if name == "__TEXT" || name == "__text" {
                Some(seg.address())
            } else {
                None
            }
        })
        .next()
        .or_else(|| file.segments().map(|s| s.address()).min())
        .unwrap_or(0)
}

fn strip_hash_suffix(s: &str) -> &str {
    if let Some(idx) = s.rfind("::h") {
        let suffix = &s[idx + 3..];
        if !suffix.is_empty() && suffix.len() <= 16 && suffix.bytes().all(|b| b.is_ascii_hexdigit())
        {
            return &s[..idx];
        }
    }
    s
}

// v0 mangling demangles inherent impl methods as `<path::Type>::method`; rewrite
// to `path::Type::method` so they match registered instrumented names. Trait impl
// symbols (`<Type as Trait>::method`) are left untouched and stay unmatched.
fn strip_inherent_impl_brackets(s: &str) -> Cow<'_, str> {
    let Some(rest) = s.strip_prefix('<') else {
        return Cow::Borrowed(s);
    };

    let mut depth = 1usize;
    let mut close = None;
    for (idx, ch) in rest.char_indices() {
        match ch {
            '<' => depth += 1,
            '>' => {
                depth -= 1;
                if depth == 0 {
                    close = Some(idx);
                    break;
                }
            }
            _ => {}
        }
    }

    let Some(close) = close else {
        return Cow::Borrowed(s);
    };
    let inner = &rest[..close];
    let tail = &rest[close + 1..];
    if inner.contains(" as ") || !tail.starts_with("::") {
        return Cow::Borrowed(s);
    }
    Cow::Owned(format!("{inner}{tail}"))
}

#[cfg_attr(feature = "hotpath-meta", hotpath_meta::measure)]
fn match_eligible_symbol(
    normalized: &str,
    match_to_display: &HashMap<&'static str, &'static str>,
) -> Option<&'static str> {
    if let Some(&display) = match_to_display.get(normalized) {
        return Some(display);
    }

    // A short registry is cheaper to scan than to hash every path prefix.
    if match_to_display.len() <= 32 {
        return match_to_display
            .iter()
            .filter(|(candidate, _)| {
                normalized
                    .strip_prefix(**candidate)
                    .is_some_and(|rest| rest.starts_with("::"))
            })
            .max_by_key(|(candidate, _)| candidate.len())
            .map(|(_, &display)| display);
    }

    let mut prefix = normalized;
    while let Some(end) = prefix.rfind("::") {
        if let Some(&display) = match_to_display.get(&prefix[..end]) {
            return Some(display);
        }
        // Retain the first colon so overlapping separators still match.
        prefix = &prefix[..=end];
    }
    None
}

fn visit_sample_stacks(
    stacks: &[Option<usize>],
    weights: Option<&[i64]>,
    stack_count: usize,
    weights_by_root: &mut Vec<Option<u64>>,
    mut attribute: impl FnMut(Option<usize>, u64),
) -> u64 {
    let mut total = 0;
    let roots = stacks.iter().enumerate().map(|(i, &root)| {
        let weight = weights
            .and_then(|weights| weights.get(i))
            .map_or(1, |&weight| weight.max(0) as u64);
        total += weight;
        (root, weight)
    });

    // Walk repeated roots once, but avoid a dense table for sparse profiles.
    if stack_count > 0 && stack_count <= stacks.len() / 2 {
        weights_by_root.resize(stack_count, None);
        for (root, weight) in roots {
            if let Some(cell) = root.and_then(|root| weights_by_root.get_mut(root)) {
                *cell.get_or_insert(0) += weight;
            }
        }
        for (root, weight) in weights_by_root.drain(..).enumerate() {
            if let Some(weight) = weight {
                attribute(Some(root), weight);
            }
        }
    } else {
        for (root, weight) in roots {
            attribute(root, weight);
        }
    }

    total
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use crate::lib_on::functions::cpu::json::Lib;
    use crate::lib_on::functions::cpu::samply::{
        build_lib_index, match_eligible_symbol, strip_hash_suffix, strip_inherent_impl_brackets,
    };

    fn identity_map<const N: usize>(
        items: [&'static str; N],
    ) -> HashMap<&'static str, &'static str> {
        items.into_iter().map(|s| (s, s)).collect()
    }

    #[cfg(unix)]
    #[test]
    fn reuses_symbol_index_only_for_the_same_file_and_labels() {
        use std::{fs::File, sync::Arc};

        use crate::lib_on::functions::cpu::samply::SymbolCache;

        let cache = SymbolCache::default();
        let exe = std::env::current_exe().unwrap();
        let open = || File::open(&exe).unwrap();
        let first = cache.index(open(), &identity_map(["first"])).unwrap();
        let reused = cache.index(open(), &identity_map(["first"])).unwrap();
        let changed = cache.index(open(), &identity_map(["other"])).unwrap();
        assert!(Arc::ptr_eq(&first, &reused));
        assert!(!Arc::ptr_eq(&first, &changed));
        assert!(cache
            .index(
                File::open(concat!(env!("CARGO_MANIFEST_DIR"), "/Cargo.toml")).unwrap(),
                &identity_map(["other"])
            )
            .is_none());
    }

    #[test]
    fn falls_back_to_executable_when_debug_file_is_unavailable() {
        let exe = std::env::current_exe().unwrap();
        let lib = Lib {
            path: Some(exe.to_str().unwrap().into()),
            debug_path: Some(exe.join("missing.debug").to_str().unwrap().into()),
        };
        assert!(build_lib_index(&lib, &HashMap::new()).is_some());
    }

    #[test]
    fn groups_repeated_roots_without_losing_weights() {
        use crate::lib_on::functions::cpu::samply::visit_sample_stacks;

        let stacks = [Some(0), Some(0), None, Some(1), Some(9), Some(0)];
        let sample_weights = [0, 5, 7, 0, 11];
        let mut scratch = Vec::new();
        for stack_count in [8, 2, 2] {
            let mut weights = HashMap::new();
            let mut visits = 0;
            let total = visit_sample_stacks(
                &stacks,
                Some(&sample_weights),
                stack_count,
                &mut scratch,
                |root, weight| {
                    if let Some(root) = root.filter(|&root| root < stack_count) {
                        *weights.entry(root).or_insert(0) += weight;
                        visits += 1;
                    }
                },
            );
            assert_eq!(total, 24);
            assert_eq!(weights, HashMap::from([(0, 6), (1, 0)]));
            assert_eq!(visits, if stack_count == 2 { 2 } else { 4 });
        }
    }

    #[test]
    fn strips_rust_hash_suffix() {
        assert_eq!(
            strip_hash_suffix("mevlog::main::h4096f6e9269ba5f4"),
            "mevlog::main"
        );
        assert_eq!(strip_hash_suffix("mevlog::main"), "mevlog::main");
    }

    #[test]
    fn strips_v0_inherent_impl_brackets() {
        assert_eq!(
            strip_inherent_impl_brackets("<cpu_symbols::Worker>::method_heavy_work"),
            "cpu_symbols::Worker::method_heavy_work"
        );
        assert_eq!(
            strip_inherent_impl_brackets("<foo::Bar<u32>>::method::{closure#0}"),
            "foo::Bar<u32>::method::{closure#0}"
        );
        assert_eq!(
            strip_inherent_impl_brackets("<foo::Bar as core::fmt::Debug>::fmt"),
            "<foo::Bar as core::fmt::Debug>::fmt"
        );
        assert_eq!(
            strip_inherent_impl_brackets("cpu_symbols::free_heavy_work"),
            "cpu_symbols::free_heavy_work"
        );
        assert_eq!(
            strip_inherent_impl_brackets("<unclosed::Bracket"),
            "<unclosed::Bracket"
        );
    }

    #[test]
    fn matches_v0_inherent_impl_method() {
        let eligible = identity_map(["cpu_symbols::Worker::method_heavy_work"]);

        let normalized = strip_inherent_impl_brackets("<cpu_symbols::Worker>::method_heavy_work");
        let matched = match_eligible_symbol(&normalized, &eligible);

        assert_eq!(matched, Some("cpu_symbols::Worker::method_heavy_work"));
    }

    #[test]
    fn matches_async_closure_symbol_by_prefix() {
        let eligible = identity_map([
            "mevlog::misc::rpc_tracing::rpc_tx_calls",
            "mevlog::misc::shared_init::init_deps",
        ]);

        let matched = match_eligible_symbol(
            "mevlog::misc::rpc_tracing::rpc_tx_calls::{{closure}}::{{closure}}",
            &eligible,
        );

        assert_eq!(matched, Some("mevlog::misc::rpc_tracing::rpc_tx_calls"));
    }

    #[test]
    fn prefers_longest_prefix_match() {
        let eligible = identity_map(["mevlog::misc", "mevlog::misc::rpc_tracing::rpc_tx_calls"]);

        let matched = match_eligible_symbol(
            "mevlog::misc::rpc_tracing::rpc_tx_calls::{{closure}}",
            &eligible,
        );

        assert_eq!(matched, Some("mevlog::misc::rpc_tracing::rpc_tx_calls"));
        let mut large = eligible;
        large.extend(
            "a b c d e f g h i j k l m n o p q r s t u v w x y z aa bb cc dd ee ff gg"
                .split_whitespace()
                .map(|name| (name, name)),
        );
        assert_eq!(
            match_eligible_symbol("mevlog::misc::rpc_tracing::rpc_tx_calls:::suffix", &large),
            matched
        );
    }

    #[test]
    fn label_alias_resolves_to_display_name() {
        let mut map: HashMap<&'static str, &'static str> = HashMap::new();
        map.insert("mevlog::compute_hash", "custom_label");

        let matched = match_eligible_symbol("mevlog::compute_hash::{{closure}}", &map);

        assert_eq!(matched, Some("custom_label"));
        assert_eq!(
            match_eligible_symbol("mevlog::compute_hash:::suffix", &map),
            matched
        );
        assert_eq!(match_eligible_symbol("mevlog::compute_hash", &map), matched);
        assert_eq!(
            match_eligible_symbol("mevlog::compute_hashes::{{closure}}", &map),
            None
        );
    }
}
