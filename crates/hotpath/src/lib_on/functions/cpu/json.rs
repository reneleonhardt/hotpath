use serde::Deserialize;

use crate::json::CpuSampleUnit;

#[derive(Deserialize)]
pub(crate) struct Profile {
    #[serde(default)]
    meta: ProfileMeta,
    #[serde(default)]
    pub(crate) libs: Vec<Lib>,
    #[serde(default)]
    pub(crate) threads: Vec<Thread>,
}

#[derive(Deserialize, Default)]
struct ProfileMeta {
    #[serde(rename = "sampleUnits", default)]
    sample_units: SampleUnits,
}

#[derive(Deserialize, Default)]
struct SampleUnits {
    #[serde(rename = "threadCPUDelta", default)]
    cpu_delta: Option<String>,
}

impl Profile {
    pub(crate) fn profiled_threads(
        &self,
        pid: u32,
    ) -> (impl Iterator<Item = &Thread> + Clone, bool) {
        // Samply gives earlier execs of the same PID lower generations.
        let generation = self
            .threads
            .iter()
            .filter_map(|thread| thread.process_generation(pid))
            .max();
        (
            self.threads.iter().filter(move |thread| {
                generation.is_none() || thread.process_generation(pid) == generation
            }),
            generation.is_some(),
        )
    }

    pub(crate) fn sample_unit<'a>(
        &self,
        mut threads: impl Iterator<Item = &'a Thread> + Clone,
    ) -> CpuSampleUnit {
        let has_complete_deltas = threads.clone().all(|thread| {
            let samples = &thread.samples;
            samples.stack.is_empty()
                || samples
                    .thread_cpu_delta
                    .as_ref()
                    .is_some_and(|deltas| deltas.len() == samples.stack.len())
        });
        let has_cpu_time = has_complete_deltas
            && threads.any(|thread| {
                !thread.samples.stack.is_empty()
                    && thread
                        .samples
                        .thread_cpu_delta
                        .iter()
                        .flatten()
                        .any(|&delta| delta > 0)
            });

        if has_cpu_time {
            match self.meta.sample_units.cpu_delta.as_deref() {
                Some("µs") => CpuSampleUnit::Microseconds,
                Some("ns") => CpuSampleUnit::Nanoseconds,
                _ => CpuSampleUnit::Unknown,
            }
        } else {
            CpuSampleUnit::Samples
        }
    }
}

#[derive(Deserialize, Default)]
pub(crate) struct Lib {
    #[serde(default)]
    pub(crate) path: Option<String>,
    #[serde(rename = "debugPath", default)]
    pub(crate) debug_path: Option<String>,
    #[serde(rename = "codeId", default)]
    pub(crate) code_id: Option<String>,
}

#[derive(Deserialize)]
pub(crate) struct Thread {
    #[serde(default)]
    pid: Option<Pid>,
    pub(crate) samples: Samples,
    #[serde(rename = "stackTable")]
    pub(crate) stack_table: StackTable,
    #[serde(rename = "frameTable")]
    pub(crate) frame_table: FrameTable,
    #[serde(rename = "funcTable")]
    pub(crate) func_table: FuncTable,
    #[serde(rename = "resourceTable")]
    pub(crate) resource_table: ResourceTable,
}

#[derive(Deserialize)]
#[serde(untagged)]
enum Pid {
    Numeric(u32),
    Text(String),
}

impl Thread {
    pub(crate) fn process_generation(&self, pid: u32) -> Option<u32> {
        match self.pid.as_ref()? {
            Pid::Numeric(value) => (*value == pid).then_some(0),
            Pid::Text(value) => {
                let (id, generation) = value.split_once('.').unwrap_or((value.as_str(), "0"));
                if id.parse::<u32>().ok()? != pid {
                    return None;
                }
                generation.parse().ok()
            }
        }
    }
}

#[derive(Deserialize)]
pub(crate) struct Samples {
    #[serde(default)]
    pub(crate) stack: Vec<Option<usize>>,
    #[serde(default)]
    pub(crate) weight: Option<Vec<i64>>,
    #[serde(rename = "weightType", default)]
    weight_type: Option<String>,
    #[serde(rename = "threadCPUDelta", default)]
    pub(crate) thread_cpu_delta: Option<Vec<i64>>,
}

impl Samples {
    pub(crate) fn weights(&self, unit: CpuSampleUnit) -> Option<&[i64]> {
        if unit != CpuSampleUnit::Samples {
            self.thread_cpu_delta.as_deref()
        } else if self
            .weight_type
            .as_deref()
            .is_none_or(|kind| kind == "samples")
        {
            self.weight.as_deref()
        } else {
            None
        }
    }
}

#[derive(Deserialize)]
pub(crate) struct StackTable {
    #[serde(default)]
    pub(crate) prefix: Vec<Option<usize>>,
    #[serde(default)]
    pub(crate) frame: Vec<usize>,
}

#[derive(Deserialize)]
pub(crate) struct FrameTable {
    #[serde(default)]
    pub(crate) address: Vec<i64>,
    #[serde(default)]
    pub(crate) func: Vec<usize>,
}

#[derive(Deserialize)]
pub(crate) struct FuncTable {
    #[serde(default)]
    pub(crate) resource: Vec<i64>,
}

#[derive(Deserialize)]
pub(crate) struct ResourceTable {
    #[serde(default)]
    pub(crate) lib: Vec<Option<i64>>,
}

#[cfg(test)]
mod tests {
    use crate::json::CpuSampleUnit;
    use crate::lib_on::functions::cpu::json::Profile;

    #[test]
    fn recognizes_samply_process_ids() {
        use crate::lib_on::functions::cpu::json::Thread;

        for (pid, generation) in [
            (serde_json::json!("42"), Some(0)),
            (serde_json::json!(42), Some(0)),
            (serde_json::json!("42.1"), Some(1)),
            (serde_json::json!("42.10"), Some(10)),
            (serde_json::json!("43.1"), None),
            (serde_json::json!("42.invalid"), None),
            (serde_json::Value::Null, None),
        ] {
            let thread: Thread = serde_json::from_value(serde_json::json!({
                "pid": pid, "samples": {}, "stackTable": {},
                "frameTable": {}, "funcTable": {}, "resourceTable": {}
            }))
            .unwrap();
            assert_eq!(thread.process_generation(42), generation);
        }
    }

    #[test]
    fn uses_only_the_latest_matching_process_generation() {
        let thread = |pid, deltas| {
            serde_json::json!({
                "pid": pid, "samples": {"stack": [null], "threadCPUDelta": deltas},
                "stackTable": {}, "frameTable": {}, "funcTable": {}, "resourceTable": {},
            })
        };
        let profile: Profile = serde_json::from_value(serde_json::json!({
            "meta": {"sampleUnits": {"threadCPUDelta": "µs"}},
            "threads": [thread("42", None), thread("84", None), thread("42.1", Some(vec![7]))],
        }))
        .unwrap();
        let (threads, found) = profile.profiled_threads(42);
        assert!(found);
        assert_eq!(
            profile.sample_unit(threads.clone()),
            CpuSampleUnit::Microseconds
        );
        assert_eq!(threads.count(), 1);
        let (threads, found) = profile.profiled_threads(5);
        assert!(!found);
        assert_eq!(profile.sample_unit(threads.clone()), CpuSampleUnit::Samples);
        assert_eq!(threads.count(), 3);
    }

    #[test]
    fn selects_one_weight_unit_for_the_whole_profile() {
        let thread = |delta, weight_type| {
            serde_json::json!({
                "samples": {
                    "stack": [null, null],
                    "threadCPUDelta": delta,
                    "weight": [3, 5],
                    "weightType": weight_type,
                },
                "stackTable": {}, "frameTable": {}, "funcTable": {}, "resourceTable": {}
            })
        };
        let profile = |threads| {
            serde_json::from_value::<Profile>(serde_json::json!({
                "meta": {"sampleUnits": {"threadCPUDelta": "µs"}},
                "threads": threads
            }))
            .unwrap()
        };
        let timed = thread(vec![0, 19], "samples");
        let zero = thread(vec![0, 0], "samples");
        let complete = profile(vec![timed.clone(), zero.clone()]);
        let unit = complete.sample_unit(complete.threads.iter());
        assert_eq!(unit, CpuSampleUnit::Microseconds);
        assert_eq!(complete.threads[1].samples.weights(unit), Some(&[0, 0][..]));
        let incomplete = profile(vec![timed, thread(vec![17], "bytes")]);
        let unit = incomplete.sample_unit(incomplete.threads.iter());
        assert_eq!(unit, CpuSampleUnit::Samples);
        assert_eq!(
            incomplete.threads[0].samples.weights(unit),
            Some(&[3, 5][..])
        );
        assert_eq!(incomplete.threads[1].samples.weights(unit), None);
        let zero = profile(vec![zero]);
        assert_eq!(
            zero.sample_unit(zero.threads.iter()),
            CpuSampleUnit::Samples
        );
    }
}
