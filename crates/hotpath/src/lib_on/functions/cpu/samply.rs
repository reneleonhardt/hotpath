use std::borrow::Cow;
use std::collections::{HashMap, HashSet, VecDeque};
use std::fmt::Write as _;
use std::fs::{File, Metadata};
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
    final_names: Option<HashMap<&'static str, u32>>,
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

    let Some(display_to_id) =
        final_names.or_else(crate::lib_on::functions::get_instrumented_names_and_ids)
    else {
        warn!("cpu report: instrumented function registry unavailable; skipping CPU report");
        return None;
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
        warn!("cpu report: could not resolve current_exe; only the current process will be attributed");
    }

    let mut libraries: Vec<_> = (0..profile.libs.len()).map(|_| None).collect();
    let primary = profile.libs.iter().enumerate().find_map(|(i, lib)| {
        let is_primary = primary_lib_name.as_deref().is_some_and(|name| {
            lib.path
                .iter()
                .chain(&lib.debug_path)
                .any(|path| Path::new(path).file_name() == Some(name))
        });
        if !is_primary {
            return None;
        }
        let entry = libraries[i].insert(ReportLibrary::load(lib, &match_to_display));
        entry.as_ref().map(|_| i)
    });

    let mut sample_counts: HashMap<&'static str, u64> = HashMap::new();
    let mut total_samples: u64 = 0;
    let mut attributed_samples: u64 = 0;
    let inclusive = *CPU_INCLUSIVE;
    let mut matched = HashSet::new();
    let mut weights_by_root = Vec::new();

    let (threads, process_found) = profile.profiled_threads(std::process::id());
    let sample_unit = profile.sample_unit(threads.clone());
    for thread in threads {
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

                if let Some(lib) =
                    lib_opt.filter(|&lib| address >= 0 && (process_found || Some(lib) == primary))
                {
                    if let Some(library) = libraries.get_mut(lib).and_then(|entry| {
                        entry
                            .get_or_insert_with(|| {
                                ReportLibrary::load(&profile.libs[lib], &match_to_display)
                            })
                            .as_mut()
                    }) {
                        let address = address as u64;
                        let eligible = |name: &&str| {
                            Some(lib) == primary
                                || label_to_symbol
                                    .get(*name)
                                    .copied()
                                    .unwrap_or(*name)
                                    .contains("::")
                        };
                        let symbol = library.index.lookup(address).filter(eligible);
                        #[cfg(feature = "hotpath-cpu-inline")]
                        let symbol = {
                            let direct = library.index.direct_inline.as_ref();
                            #[cfg(target_os = "macos")]
                            let direct = direct
                                .or_else(|| library.adjacent.as_ref()?.direct_inline.as_ref());
                            let labels = library.inline.lookup(
                                library._path,
                                address,
                                library.index.image_base,
                                direct,
                                &match_to_display,
                            );
                            if inclusive {
                                matched.extend(labels.iter().copied().filter(eligible));
                                symbol
                            } else {
                                labels.iter().copied().find(eligible).or(symbol)
                            }
                        };
                        if let Some(sym) = symbol {
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

    let total_matches: usize = libraries
        .into_iter()
        .flatten()
        .flatten()
        .map(|library| library.index.ranges.len())
        .sum();
    info!(
        "cpu report: total_samples={total_samples} attributed_samples={attributed_samples} matched_symbols={total_matches} stats_rows={}",
        stats.len()
    );

    if attributed_samples == 0 {
        if total_matches == 0 {
            warn!(
                "cpu report: no instrumented symbols found in any sampled library - ensure the binary was built with debug symbols and not stripped"
            );
        }
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
    #[cfg(feature = "hotpath-cpu-inline")]
    image_base: u64,
    #[cfg(feature = "hotpath-cpu-inline")]
    direct_inline: Option<Mutex<DirectInlineSymbols>>,
    ranges: Vec<(u64, u64, &'static str)>,
    code_id: Option<Box<[u8]>>,
}

#[cfg_attr(feature = "hotpath-meta", hotpath_meta::measure_all)]
impl LibSymbolIndex {
    fn lookup(&self, addr: u64) -> Option<&'static str> {
        let idx = self.ranges.partition_point(|(start, _, _)| *start <= addr);
        let (_, end, sym) = self.ranges[idx.checked_sub(1)?];
        (addr < end).then_some(sym)
    }
}

fn matches_code_id(actual: Option<&[u8]>, recorded: Option<&str>) -> bool {
    let Some((actual, recorded)) = actual.zip(recorded.filter(|id| !id.is_empty())) else {
        return true;
    };
    const HEX: &[u8; 16] = b"0123456789abcdef";
    recorded.len() == actual.len() * 2
        && actual
            .iter()
            .zip(recorded.as_bytes().chunks_exact(2))
            .all(|(&byte, pair)| {
                pair[0].to_ascii_lowercase() == HEX[(byte >> 4) as usize]
                    && pair[1].to_ascii_lowercase() == HEX[(byte & 0xf) as usize]
            })
}

type FileStamp = (u64, u64, u64, i64, i64);

#[cfg(unix)]
fn file_stamp(metadata: &Metadata) -> Option<FileStamp> {
    use std::os::unix::fs::MetadataExt;

    Some((
        metadata.dev(),
        metadata.ino(),
        metadata.len(),
        metadata.ctime(),
        metadata.ctime_nsec(),
    ))
}

#[cfg(not(unix))]
fn file_stamp(_: &Metadata) -> Option<FileStamp> {
    None
}

#[derive(Default)]
struct CachedSymbols {
    labels: HashMap<&'static str, &'static str>,
    indexes: VecDeque<(FileStamp, Option<Arc<LibSymbolIndex>>)>,
}

#[derive(Default)]
struct SymbolCache(Mutex<CachedSymbols>);

impl SymbolCache {
    fn index(
        &self,
        path: &Path,
        labels: &HashMap<&'static str, &'static str>,
        recorded_code_id: Option<&str>,
    ) -> Option<Arc<LibSymbolIndex>> {
        let metadata = std::fs::metadata(path).ok()?;
        if !metadata.is_file() {
            return None;
        }
        let stamp = file_stamp(&metadata);
        if let (Some(stamp), Ok(mut cache)) = (stamp, self.0.lock()) {
            if &cache.labels == labels {
                if let Some(pos) = cache.indexes.iter().position(|(saved, _)| *saved == stamp) {
                    let index = cache.indexes[pos].1.as_ref();
                    if index.is_some_and(|index| {
                        !matches_code_id(index.code_id.as_deref(), recorded_code_id)
                    }) {
                        return None;
                    }
                    let entry = cache.indexes.remove(pos)?;
                    let index = entry.1.clone();
                    cache.indexes.push_back(entry);
                    return index;
                }
            }
        }

        let mut file = File::open(path).ok()?;
        let stamp = file.metadata().ok().as_ref().and_then(file_stamp);
        let mut bytes = Vec::new();
        file.read_to_end(&mut bytes).ok()?;
        let index = match object::File::parse(&*bytes) {
            Ok(parsed) => Some(Arc::new(build_symbol_index(
                &parsed,
                labels,
                recorded_code_id,
            )?)),
            Err(_) => None,
        };
        if let Some(stamp) =
            stamp.filter(|&stamp| Some(stamp) == file.metadata().ok().as_ref().and_then(file_stamp))
        {
            if let Ok(mut cache) = self.0.lock() {
                if &cache.labels != labels {
                    cache.labels.clone_from(labels);
                    cache.indexes.clear();
                }
                const MAX_INDEXES: usize = 32;
                if cache.indexes.len() == MAX_INDEXES {
                    cache.indexes.pop_front();
                }
                cache.indexes.push_back((stamp, index.clone()));
            }
        }
        index
    }
}

static SYMBOL_CACHE: std::sync::LazyLock<SymbolCache> =
    std::sync::LazyLock::new(SymbolCache::default);

#[cfg_attr(feature = "hotpath-meta", hotpath_meta::measure)]
fn build_lib_index<'a>(
    lib: &'a Lib,
    match_to_display: &HashMap<&'static str, &'static str>,
) -> Option<(&'a str, Arc<LibSymbolIndex>)> {
    let debug_path = lib.debug_path.as_deref().filter(|p| !p.is_empty());
    let binary_path = lib.path.as_deref().filter(|&path| Some(path) != debug_path);
    debug_path.into_iter().chain(binary_path).find_map(|path| {
        let index =
            SYMBOL_CACHE.index(Path::new(path), match_to_display, lib.code_id.as_deref())?;
        Some((path, index))
    })
}

struct ReportLibrary<'a> {
    _path: &'a str,
    index: Arc<LibSymbolIndex>,
    #[cfg(feature = "hotpath-cpu-inline")]
    inline: InlineSymbols,
    #[cfg(all(feature = "hotpath-cpu-inline", target_os = "macos"))]
    adjacent: Option<Arc<LibSymbolIndex>>,
}

impl<'a> ReportLibrary<'a> {
    fn load(lib: &'a Lib, labels: &HashMap<&'static str, &'static str>) -> Option<Self> {
        let (path, index) = build_lib_index(lib, labels)?;
        Some(Self {
            #[cfg(all(feature = "hotpath-cpu-inline", target_os = "macos"))]
            adjacent: if index.direct_inline.is_none() && lib.path.as_deref() == Some(path) {
                adjacent_dsym(path, lib.code_id.as_deref(), labels)
            } else {
                None
            },
            _path: path,
            #[cfg(feature = "hotpath-cpu-inline")]
            inline: InlineSymbols::default(),
            index,
        })
    }
}

#[cfg(all(feature = "hotpath-cpu-inline", target_os = "macos"))]
fn adjacent_dsym(
    path: &str,
    code_id: Option<&str>,
    labels: &HashMap<&'static str, &'static str>,
) -> Option<Arc<LibSymbolIndex>> {
    let code_id = code_id.filter(|id| !id.is_empty())?;
    let name = Path::new(path).file_name()?;
    let mut bundle = std::ffi::OsString::from(path);
    bundle.push(".dSYM");
    let source = std::path::PathBuf::from(bundle)
        .join("Contents/Resources/DWARF")
        .join(name);
    SYMBOL_CACHE
        .index(&source, labels, Some(code_id))
        .filter(|index| index.code_id.is_some() && index.direct_inline.is_some())
}

fn build_symbol_index(
    parsed: &object::File<'_>,
    match_to_display: &HashMap<&'static str, &'static str>,
    recorded_code_id: Option<&str>,
) -> Option<LibSymbolIndex> {
    let code_id = parsed
        .mach_uuid()
        .ok()
        .flatten()
        .map(|id| Box::from(id) as Box<[u8]>)
        .or_else(|| parsed.build_id().ok().flatten().map(Box::from));
    if !matches_code_id(code_id.as_deref(), recorded_code_id) {
        return None;
    }

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

    Some(LibSymbolIndex {
        #[cfg(feature = "hotpath-cpu-inline")]
        image_base: base,
        #[cfg(feature = "hotpath-cpu-inline")]
        direct_inline: DirectInlineSymbols::new(parsed).map(Mutex::new),
        ranges,
        code_id,
    })
}

#[cfg(feature = "hotpath-cpu-inline")]
type InlineLabels = Option<Arc<[&'static str]>>;

#[cfg(feature = "hotpath-cpu-inline")]
struct DirectInlineSymbols {
    context: addr2line::Context<gimli::EndianArcSlice<gimli::RunTimeEndian>>,
    by_address: HashMap<u64, InlineLabels>,
}

#[cfg(feature = "hotpath-cpu-inline")]
impl DirectInlineSymbols {
    fn new(file: &object::File<'_>) -> Option<Self> {
        use object::{read::macho::MachHeader, ObjectSection};

        // Mach-O executables may resolve through a dSYM or external object files.
        match file {
            object::File::Elf32(_) | object::File::Elf64(_) => {}
            object::File::MachO64(file)
                if file.macho_header().filetype(file.endian()) == object::macho::MH_DSYM => {}
            _ => return None,
        }
        file.section_by_name(".debug_info")?;
        let endian = if file.is_little_endian() {
            gimli::RunTimeEndian::Little
        } else {
            gimli::RunTimeEndian::Big
        };
        let mut remaining = 4 * 1024 * 1024_u64;
        let mut dwarf = gimli::Dwarf::load(|id| {
            let Some(section) = file.section_by_name(id.name()) else {
                return Ok(gimli::EndianArcSlice::new(Arc::from([]), endian));
            };
            let data = section.compressed_data().map_err(|_| ())?;
            remaining = remaining.checked_sub(data.uncompressed_size).ok_or(())?;
            let bytes = data.decompress().map_err(|_| ())?;
            Ok::<_, ()>(gimli::EndianArcSlice::new(
                Arc::from(bytes.as_ref()),
                endian,
            ))
        })
        .ok()?;
        dwarf.populate_abbreviations_cache(gimli::AbbreviationsCacheStrategy::Duplicates);
        Some(Self {
            context: addr2line::Context::from_dwarf(dwarf).ok()?,
            by_address: HashMap::new(),
        })
    }

    fn lookup(
        &mut self,
        address: u64,
        labels: &HashMap<&'static str, &'static str>,
        demangled: &mut String,
    ) -> Option<InlineLabels> {
        if let Some(names) = self.by_address.get(&address) {
            return Some(names.clone());
        }
        // The fallback loader resolves DWO/DWP references without caching them across reports.
        let addr2line::LookupResult::Output(frames) = self.context.find_frames(address) else {
            return None;
        };
        let names = collect_inline_labels(frames.ok()?, labels, demangled).ok()?;
        const MAX_INLINE_ADDRESSES: usize = 4096;
        if self.by_address.len() < MAX_INLINE_ADDRESSES {
            self.by_address.insert(address, names.clone());
        }
        Some(names)
    }
}

#[cfg(feature = "hotpath-cpu-inline")]
#[derive(Default)]
struct InlineSymbols {
    fallback: Option<Option<addr2line::Loader>>,
    by_address: HashMap<u64, InlineLabels>,
    demangled: String,
}

#[cfg(feature = "hotpath-cpu-inline")]
impl InlineSymbols {
    fn lookup(
        &mut self,
        path: &str,
        address: u64,
        base: u64,
        direct: Option<&Mutex<DirectInlineSymbols>>,
        labels: &HashMap<&'static str, &'static str>,
    ) -> &[&'static str] {
        self.by_address
            .entry(address)
            .or_insert_with(|| {
                let address = base.checked_add(address)?;
                if let Some(names) = direct.and_then(|direct| {
                    direct
                        .try_lock()
                        .ok()?
                        .lookup(address, labels, &mut self.demangled)
                }) {
                    return names;
                }
                let loader = self
                    .fallback
                    .get_or_insert_with(|| addr2line::Loader::new(path).ok())
                    .as_ref()?;
                let frames = loader.find_frames(address).ok()?;
                collect_inline_labels(frames, labels, &mut self.demangled)
                    .ok()
                    .flatten()
            })
            .as_deref()
            .unwrap_or_default()
    }
}

#[cfg(feature = "hotpath-cpu-inline")]
fn collect_inline_labels<R: gimli::Reader>(
    mut frames: addr2line::FrameIter<'_, R>,
    labels: &HashMap<&'static str, &'static str>,
    demangled: &mut String,
) -> Result<InlineLabels, gimli::Error> {
    let mut found = Vec::new();
    while let Some(frame) = frames.next()? {
        let Some(raw) = frame
            .function
            .as_ref()
            .and_then(|name| name.raw_name().ok())
        else {
            continue;
        };
        demangled.clear();
        write!(demangled, "{:#}", rustc_demangle::demangle(&raw)).unwrap();
        let normalized = strip_inherent_impl_brackets(strip_hash_suffix(demangled));
        if let Some(label) = match_eligible_symbol(&normalized, labels) {
            found.push(label);
        }
    }
    Ok((!found.is_empty()).then(|| Arc::from(found)))
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
        use std::{path::Path, sync::Arc};

        use crate::lib_on::functions::cpu::samply::SymbolCache;

        let cache = SymbolCache::default();
        let exe = std::env::current_exe().unwrap();
        let first = cache.index(&exe, &identity_map(["first"]), None).unwrap();
        let reused = cache.index(&exe, &identity_map(["first"]), None).unwrap();
        let changed = cache.index(&exe, &identity_map(["other"]), None).unwrap();
        assert!(Arc::ptr_eq(&first, &reused));
        assert!(!Arc::ptr_eq(&first, &changed));
        assert!(cache
            .index(
                Path::new(concat!(env!("CARGO_MANIFEST_DIR"), "/Cargo.toml")),
                &identity_map(["other"]),
                None,
            )
            .is_none());
    }

    #[test]
    fn rejects_a_different_recorded_code_id() {
        use crate::lib_on::functions::cpu::samply::matches_code_id;

        let actual = Some(&[0xab, 0x13][..]);
        assert!(matches_code_id(actual, Some("Ab13")));
        assert!(matches_code_id(actual, None));
        assert!(!matches_code_id(actual, Some("AB12")));
        assert!(!matches_code_id(actual, Some("AB")));
    }

    #[test]
    fn falls_back_to_executable_when_debug_file_is_unavailable() {
        let exe = std::env::current_exe().unwrap();
        let lib = Lib {
            path: Some(exe.to_str().unwrap().into()),
            debug_path: Some(exe.join("missing.debug").to_str().unwrap().into()),
            ..Lib::default()
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
    fn final_cpu_report_uses_supplied_function_names() {
        let path =
            std::env::temp_dir().join(format!("hotpath-final-cpu-{}.json", std::process::id()));
        std::fs::write(&path, "{}").unwrap();
        let report = crate::functions::cpu::build_cpu_report_from_path(
            "run",
            &path,
            Some(HashMap::from([("work", 1)])),
        );
        std::fs::remove_file(path).unwrap();
        assert_eq!(report.unwrap().caller_name, "run");
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
