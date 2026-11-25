use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::sync::LazyLock;

use crate::instant::Instant;
#[cfg(feature = "hotpath-sink")]
use crate::lib_on::functions::sink::SinkScope;
#[cfg(feature = "hotpath-sink")]
use crate::ScopeKind;

#[derive(Debug, Default)]
pub(crate) struct AsyncAllocBridge {
    bytes_total: AtomicU64,
    count_total: AtomicU64,
}

impl AsyncAllocBridge {
    #[inline]
    pub(crate) fn add(&self, bytes: u64, count: u64) {
        self.bytes_total.fetch_add(bytes, Ordering::Relaxed);
        self.count_total.fetch_add(count, Ordering::Relaxed);
    }

    #[inline]
    pub(crate) fn snapshot(&self) -> (Option<u64>, Option<u64>) {
        (
            Some(self.bytes_total.load(Ordering::Relaxed)),
            Some(self.count_total.load(Ordering::Relaxed)),
        )
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum AllocMetric {
    Bytes,
    Count,
}

pub(crate) static ALLOC_METRIC: LazyLock<AllocMetric> =
    LazyLock::new(|| match std::env::var("HOTPATH_ALLOC_METRIC") {
        Ok(v) => match v.to_lowercase().as_str() {
            "bytes" => AllocMetric::Bytes,
            "count" => AllocMetric::Count,
            other => panic!(
                "Invalid HOTPATH_ALLOC_METRIC value: '{}'. Expected 'bytes' or 'count'.",
                other
            ),
        },
        Err(_) => AllocMetric::Bytes,
    });

pub(crate) static ALLOC_CUMULATIVE: LazyLock<bool> =
    LazyLock::new(|| crate::shared::env_flag("HOTPATH_ALLOC_CUMULATIVE"));

#[inline]
pub(crate) fn push_alloc_stack() {
    crate::functions::alloc::core::ALLOCATIONS.with(|stack| {
        let max_depth = crate::functions::alloc::core::MAX_DEPTH;
        let depth = stack.depth.get() as usize + 1;
        // Checked before `depth` moves: the panic allocates, and
        // `track_alloc` must still index a valid frame.
        if depth > max_depth {
            panic!(
                "hotpath: more than {max_depth} nested measured functions on one thread in alloc mode. \
                 Raise the limit by rebuilding with HOTPATH_ALLOC_MAX_DEPTH set (e.g. HOTPATH_ALLOC_MAX_DEPTH=256)."
            );
        }
        stack.depth.set(depth as u32);
        stack.elements[depth].bytes_total.set(0);
        stack.elements[depth].count_total.set(0);
    });
}

#[inline]
pub(crate) fn pop_alloc_stack() -> (u64, u64) {
    crate::functions::alloc::core::ALLOCATIONS.with(|stack| {
        assert!(stack.depth.get() > 0, "pop_alloc_stack called with depth 0");
        let depth = stack.depth.get() as usize;
        let bytes = stack.elements[depth].bytes_total.get();
        let count = stack.elements[depth].count_total.get();

        let parent = depth - 1;
        stack.depth.set(parent as u32);

        if *ALLOC_CUMULATIVE {
            if parent == 0 {
                crate::functions::alloc::state::record_root_allocations(bytes, count);
            } else {
                let parent = &stack.elements[parent];
                parent.bytes_total.set(parent.bytes_total.get() + bytes);
                parent.count_total.set(parent.count_total.get() + count);
            }
        }

        (bytes, count)
    })
}

#[inline]
#[allow(clippy::too_many_arguments)]
fn send_alloc_measurement(
    name: &'static str,
    bytes_total: Option<u64>,
    count_total: Option<u64>,
    duration_ns: Option<u64>,
    elapsed_since_start_ns: u64,
    wrapper: bool,
    tid: Option<u64>,
) {
    crate::functions::alloc::state::send_alloc_measurement(
        name,
        bytes_total,
        count_total,
        duration_ns,
        elapsed_since_start_ns,
        wrapper,
        tid,
    );
}

#[inline]
#[allow(clippy::too_many_arguments)]
fn send_alloc_measurement_with_log(
    name: &'static str,
    bytes_total: Option<u64>,
    count_total: Option<u64>,
    duration_ns: Option<u64>,
    elapsed_since_start_ns: u64,
    wrapper: bool,
    tid: Option<u64>,
    result_log: Option<String>,
) {
    crate::functions::alloc::state::send_alloc_measurement_with_log(
        name,
        bytes_total,
        count_total,
        duration_ns,
        elapsed_since_start_ns,
        wrapper,
        tid,
        result_log,
    );
}

/// Wrapper guards are never sampled - their exact total is the `%` denominator.
/// Unsampled guards keep exact allocation tracking; only the duration is skipped.
#[inline]
fn sampled_start(wrapper: bool, skipped: bool) -> Option<Instant> {
    if skipped {
        return None;
    }
    if wrapper || crate::lib_on::sampling::functions_should_time() {
        Some(Instant::now())
    } else {
        None
    }
}

#[must_use = "guard is dropped immediately without measuring anything"]
pub struct MeasurementGuardSync {
    name: &'static str,
    wrapper: bool,
    tid: u64,
    paused: bool,
    migrated: bool,
    bytes: u64,
    count: u64,
    start: Option<Instant>,
    skipped: bool,
    caller_scoped: bool,
    #[cfg(feature = "hotpath-sink")]
    sink: Option<SinkScope>,
}

impl MeasurementGuardSync {
    #[inline]
    pub fn new(name: &'static str, wrapper: bool, skipped: bool) -> Self {
        Self::build(name, wrapper, skipped, false)
    }

    /// Also registers `name` on the thread-local caller stack for SQL/HTTP
    /// source attribution (skipped for wrapper guards).
    #[inline]
    pub(crate) fn new_caller_scoped(name: &'static str, wrapper: bool, skipped: bool) -> Self {
        Self::build(name, wrapper, skipped, !wrapper && !skipped)
    }

    #[inline]
    fn build(name: &'static str, wrapper: bool, skipped: bool, caller_scoped: bool) -> Self {
        let _suspend = crate::lib_on::SuspendAllocTracking::new();
        if !skipped {
            push_alloc_stack();
        }
        if caller_scoped {
            crate::lib_on::caller_stack::push_caller(name);
        }
        // Entered before the clock read so hotpath's own timing stays inside
        // the sink scope and never includes the sink call.
        #[cfg(feature = "hotpath-sink")]
        let sink = SinkScope::enter(name, ScopeKind::of_guard(caller_scoped), skipped);

        Self {
            name,
            wrapper,
            tid: crate::tid::current_tid(),
            paused: false,
            migrated: false,
            bytes: 0,
            count: 0,
            start: sampled_start(wrapper, skipped),
            skipped,
            caller_scoped,
            #[cfg(feature = "hotpath-sink")]
            sink,
        }
    }
}

impl Drop for MeasurementGuardSync {
    #[inline]
    fn drop(&mut self) {
        if self.skipped {
            return;
        }

        let _suspend = crate::lib_on::SuspendAllocTracking::new();
        let end = Instant::now();
        #[cfg(feature = "hotpath-sink")]
        if let Some(sink) = self.sink.take() {
            sink.exit(self.name);
        }
        let duration_ns = self
            .start
            .map(|start| end.duration_since(start).as_nanos() as u64);
        let elapsed_since_start_ns = crate::lib_on::elapsed_since_start_ns(end);
        let cross_thread = crate::tid::current_tid() != self.tid;

        if self.caller_scoped && !cross_thread {
            crate::lib_on::caller_stack::pop_caller();
        }
        let (bytes_total, count_total) = if self.paused {
            (Some(self.bytes), Some(self.count))
        } else if cross_thread {
            (None, None)
        } else {
            let (bytes, count) = pop_alloc_stack();
            (Some(self.bytes + bytes), Some(self.count + count))
        };

        send_alloc_measurement(
            self.name,
            bytes_total,
            count_total,
            duration_ns,
            elapsed_since_start_ns,
            self.wrapper,
            (!self.migrated && !cross_thread).then_some(self.tid),
        );
    }
}

pin_project_lite::pin_project! {
    #[doc(hidden)]
    pub struct MeasuredAwait<F, Scopes> {
        #[pin]
        inner: F,
        scopes: Scopes,
    }
}

#[doc(hidden)]
pub fn measure_block_await<F: std::future::IntoFuture, const N: usize>(
    future: F,
    scopes: [&mut MeasurementGuardSync; N],
) -> MeasuredAwait<F::IntoFuture, [&mut MeasurementGuardSync; N]> {
    MeasuredAwait {
        inner: future.into_future(),
        scopes,
    }
}

struct PollScopes<'a, 'guard, const N: usize>(&'a mut [&'guard mut MeasurementGuardSync; N]);

impl<const N: usize> Drop for PollScopes<'_, '_, N> {
    fn drop(&mut self) {
        let _suspend = crate::lib_on::SuspendAllocTracking::new();
        for guard in self.0.iter_mut().rev() {
            if !guard.skipped && !guard.paused {
                let (bytes, count) = pop_alloc_stack();
                guard.bytes += bytes;
                guard.count += count;
                guard.paused = true;
            }
        }
    }
}

impl<F: std::future::Future, const N: usize> std::future::Future
    for MeasuredAwait<F, [&mut MeasurementGuardSync; N]>
{
    type Output = F::Output;

    fn poll(
        self: std::pin::Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Self::Output> {
        let this = self.project();
        let scopes = PollScopes(this.scopes);
        for guard in &mut *scopes.0 {
            if guard.paused {
                push_alloc_stack();
                let tid = crate::tid::current_tid();
                guard.migrated |= guard.tid != tid;
                guard.tid = tid;
                guard.paused = false;
            }
        }

        let result = this.inner.poll(cx);
        if result.is_ready() {
            // The enclosing block is still executing and owns the active frames.
            std::mem::forget(scopes);
        }
        result
    }
}

#[must_use = "guard is dropped immediately without measuring anything"]
pub struct MeasurementGuardAsync {
    name: &'static str,
    wrapper: bool,
    tid: u64,
    start: Option<Instant>,
    skipped: bool,
    alloc_bridge: Option<Arc<AsyncAllocBridge>>,
    #[cfg(feature = "hotpath-sink")]
    sink: Option<SinkScope>,
}

impl MeasurementGuardAsync {
    #[inline]
    pub(crate) fn new(
        name: &'static str,
        wrapper: bool,
        skipped: bool,
        alloc_bridge: Option<Arc<AsyncAllocBridge>>,
    ) -> Self {
        #[cfg(feature = "hotpath-sink")]
        let sink = SinkScope::enter(name, ScopeKind::Async, skipped);
        Self {
            name,
            wrapper,
            tid: crate::tid::current_tid(),
            start: sampled_start(wrapper, skipped),
            skipped,
            alloc_bridge,
            #[cfg(feature = "hotpath-sink")]
            sink,
        }
    }
}

impl Drop for MeasurementGuardAsync {
    #[inline]
    fn drop(&mut self) {
        if self.skipped {
            return;
        }

        let _suspend = crate::lib_on::SuspendAllocTracking::new();
        let end = Instant::now();
        #[cfg(feature = "hotpath-sink")]
        if let Some(sink) = self.sink.take() {
            sink.exit(self.name);
        }
        let duration_ns = self
            .start
            .map(|start| end.duration_since(start).as_nanos() as u64);
        let elapsed_since_start_ns = crate::lib_on::elapsed_since_start_ns(end);
        let (bytes_total, count_total) = self
            .alloc_bridge
            .as_ref()
            .map_or((None, None), |bridge| bridge.snapshot());

        send_alloc_measurement(
            self.name,
            bytes_total,
            count_total,
            duration_ns,
            elapsed_since_start_ns,
            self.wrapper,
            Some(self.tid),
        );
    }
}

#[must_use = "guard is dropped immediately without measuring anything"]
pub(crate) struct MeasurementGuardSyncWithLog {
    name: &'static str,
    wrapper: bool,
    tid: u64,
    start: Option<Instant>,
    finished: bool,
    skipped: bool,
    caller_scoped: bool,
    #[cfg(feature = "hotpath-sink")]
    sink: Option<SinkScope>,
}

impl MeasurementGuardSyncWithLog {
    /// Also registers `name` on the thread-local caller stack for SQL/HTTP
    /// source attribution (skipped for wrapper guards).
    #[inline]
    pub(crate) fn new_caller_scoped(name: &'static str, wrapper: bool, skipped: bool) -> Self {
        let _suspend = crate::lib_on::SuspendAllocTracking::new();
        let caller_scoped = !wrapper && !skipped;
        if !skipped {
            push_alloc_stack();
        }
        if caller_scoped {
            crate::lib_on::caller_stack::push_caller(name);
        }
        // Entered before the clock read so hotpath's own timing stays inside
        // the sink scope and never includes the sink call.
        #[cfg(feature = "hotpath-sink")]
        let sink = SinkScope::enter(name, ScopeKind::of_guard(caller_scoped), skipped);

        Self {
            name,
            wrapper,
            tid: crate::tid::current_tid(),
            start: sampled_start(wrapper, skipped),
            finished: false,
            skipped,
            caller_scoped,
            #[cfg(feature = "hotpath-sink")]
            sink,
        }
    }

    #[inline]
    pub(crate) fn finish_with_result<T: std::fmt::Debug>(mut self, result: &T) {
        self.finished = true;
        if self.skipped {
            return;
        }

        let _suspend = crate::lib_on::SuspendAllocTracking::new();
        let end = Instant::now();
        #[cfg(feature = "hotpath-sink")]
        if let Some(sink) = self.sink.take() {
            sink.exit(self.name);
        }
        let duration_ns = self
            .start
            .map(|start| end.duration_since(start).as_nanos() as u64);
        let elapsed_since_start_ns = crate::lib_on::elapsed_since_start_ns(end);
        let result_str = crate::output_on::format_debug_truncated(result);
        let cross_thread = crate::tid::current_tid() != self.tid;

        if self.caller_scoped && !cross_thread {
            crate::lib_on::caller_stack::pop_caller();
        }
        let (bytes_total, count_total) = if cross_thread {
            (None, None)
        } else {
            let (bytes, count) = pop_alloc_stack();
            (Some(bytes), Some(count))
        };

        send_alloc_measurement_with_log(
            self.name,
            bytes_total,
            count_total,
            duration_ns,
            elapsed_since_start_ns,
            self.wrapper,
            Some(self.tid),
            Some(result_str),
        );
    }
}

impl Drop for MeasurementGuardSyncWithLog {
    #[inline]
    fn drop(&mut self) {
        if self.skipped || self.finished {
            return;
        }

        let _suspend = crate::lib_on::SuspendAllocTracking::new();
        let end = Instant::now();
        #[cfg(feature = "hotpath-sink")]
        if let Some(sink) = self.sink.take() {
            sink.exit(self.name);
        }
        let duration_ns = self
            .start
            .map(|start| end.duration_since(start).as_nanos() as u64);
        let elapsed_since_start_ns = crate::lib_on::elapsed_since_start_ns(end);
        let cross_thread = crate::tid::current_tid() != self.tid;

        if self.caller_scoped && !cross_thread {
            crate::lib_on::caller_stack::pop_caller();
        }
        let (bytes_total, count_total) = if cross_thread {
            (None, None)
        } else {
            let (bytes, count) = pop_alloc_stack();
            (Some(bytes), Some(count))
        };

        send_alloc_measurement_with_log(
            self.name,
            bytes_total,
            count_total,
            duration_ns,
            elapsed_since_start_ns,
            self.wrapper,
            Some(self.tid),
            None,
        );
    }
}

#[must_use = "guard is dropped immediately without measuring anything"]
#[doc(hidden)]
pub struct MeasurementGuardAsyncWithLog {
    name: &'static str,
    wrapper: bool,
    tid: u64,
    start: Option<Instant>,
    finished: bool,
    skipped: bool,
    alloc_bridge: Option<Arc<AsyncAllocBridge>>,
    #[cfg(feature = "hotpath-sink")]
    sink: Option<SinkScope>,
}

impl MeasurementGuardAsyncWithLog {
    #[inline]
    pub(crate) fn new(
        name: &'static str,
        wrapper: bool,
        skipped: bool,
        alloc_bridge: Option<Arc<AsyncAllocBridge>>,
    ) -> Self {
        #[cfg(feature = "hotpath-sink")]
        let sink = SinkScope::enter(name, ScopeKind::Async, skipped);
        Self {
            name,
            wrapper,
            tid: crate::tid::current_tid(),
            start: sampled_start(wrapper, skipped),
            finished: false,
            skipped,
            alloc_bridge,
            #[cfg(feature = "hotpath-sink")]
            sink,
        }
    }

    #[inline]
    pub(crate) fn finish_with_result<T: std::fmt::Debug>(mut self, result: &T) {
        self.finished = true;
        if self.skipped {
            return;
        }

        let _suspend = crate::lib_on::SuspendAllocTracking::new();
        let end = Instant::now();
        #[cfg(feature = "hotpath-sink")]
        if let Some(sink) = self.sink.take() {
            sink.exit(self.name);
        }
        let duration_ns = self
            .start
            .map(|start| end.duration_since(start).as_nanos() as u64);
        let elapsed_since_start_ns = crate::lib_on::elapsed_since_start_ns(end);
        let result_str = crate::output_on::format_debug_truncated(result);
        let (bytes_total, count_total) = self
            .alloc_bridge
            .as_ref()
            .map_or((None, None), |bridge| bridge.snapshot());

        send_alloc_measurement_with_log(
            self.name,
            bytes_total,
            count_total,
            duration_ns,
            elapsed_since_start_ns,
            self.wrapper,
            Some(self.tid),
            Some(result_str),
        );
    }
}

impl Drop for MeasurementGuardAsyncWithLog {
    #[inline]
    fn drop(&mut self) {
        if self.skipped || self.finished {
            return;
        }

        let _suspend = crate::lib_on::SuspendAllocTracking::new();
        let end = Instant::now();
        #[cfg(feature = "hotpath-sink")]
        if let Some(sink) = self.sink.take() {
            sink.exit(self.name);
        }
        let duration_ns = self
            .start
            .map(|start| end.duration_since(start).as_nanos() as u64);
        let elapsed_since_start_ns = crate::lib_on::elapsed_since_start_ns(end);
        let (bytes_total, count_total) = self
            .alloc_bridge
            .as_ref()
            .map_or((None, None), |bridge| bridge.snapshot());

        send_alloc_measurement_with_log(
            self.name,
            bytes_total,
            count_total,
            duration_ns,
            elapsed_since_start_ns,
            self.wrapper,
            Some(self.tid),
            None,
        );
    }
}

#[cfg(all(test, feature = "hotpath-alloc"))]
mod tests {
    use crate::functions::alloc::core::{track_alloc, MAX_DEPTH};
    use crate::functions::alloc::guard::{pop_alloc_stack, push_alloc_stack};

    #[test]
    fn nesting_up_to_max_depth_keeps_frames_exclusive() {
        for _ in 0..MAX_DEPTH {
            push_alloc_stack();
        }
        track_alloc(64);
        assert_eq!(pop_alloc_stack(), (64, 1));
        for _ in 1..MAX_DEPTH {
            pop_alloc_stack();
        }
    }

    #[test]
    #[should_panic(expected = "HOTPATH_ALLOC_MAX_DEPTH")]
    fn nesting_past_max_depth_panics_with_hint() {
        for _ in 0..=MAX_DEPTH {
            push_alloc_stack();
        }
    }

    #[test]
    fn async_blocks_report_only_their_polled_allocations() {
        use std::alloc::{GlobalAlloc, Layout};
        use std::future::{pending, poll_fn, Future};
        use std::panic::{catch_unwind, AssertUnwindSafe};
        use std::pin::Pin;
        use std::task::{Context, Poll, Waker};

        fn allocate() {
            let alloc = hotpath::CountingAllocator::new();
            let layout = Layout::new::<[u8; 8]>();
            // SAFETY: the layout is nonzero and the same allocator frees the returned block.
            unsafe {
                let ptr = alloc.alloc(layout);
                assert!(!ptr.is_null());
                alloc.dealloc(ptr, layout);
            }
        }

        fn poll<F: Future>(future: Pin<&mut F>) -> Poll<F::Output> {
            future.poll(&mut Context::from_waker(Waker::noop()))
        }

        macro_rules! hidden_await {
            () => {
                pending::<()>().await
            };
        }

        let path =
            std::env::temp_dir().join(format!("hotpath-async-blocks-{}.json", std::process::id()));
        let profiler = hotpath::HotpathGuardBuilder::new("async blocks")
            .report("functions-alloc")
            .functions_limit(0)
            .format(hotpath::Format::Json)
            .output_path(&path)
            .build();

        drop(hotpath::measure_block!("unpolled block", async {
            allocate()
        }));

        for (resume, unwind) in [(false, false), (true, false), (true, true)] {
            let mut future = Box::pin(async move {
                hotpath::measure_block!("outer block", {
                    hotpath::measure_block!("nested block", {
                        let mut first = true;
                        poll_fn(|_| {
                            allocate();
                            if std::mem::take(&mut first) {
                                Poll::Pending
                            } else if unwind {
                                panic!("resumed poll failed");
                            } else {
                                Poll::Ready(7)
                            }
                        })
                        .await
                    })
                })
            });
            assert_eq!(poll(future.as_mut()), Poll::Pending);
            allocate();
            std::thread::spawn(move || {
                if resume {
                    if unwind {
                        assert!(catch_unwind(AssertUnwindSafe(|| poll(future.as_mut()))).is_err());
                    } else {
                        assert_eq!(poll(future.as_mut()), Poll::Ready(7));
                    }
                }
            })
            .join()
            .unwrap();
        }

        let mut hidden = Box::pin(hotpath::measure_block!("hidden await", async {
            allocate();
            hidden_await!();
        }));
        assert!(poll(hidden.as_mut()).is_pending());
        allocate();
        drop(hidden);

        let mut panics = Box::pin(hotpath::measure_block!("panicking poll", async {
            allocate();
            panic!("poll failed");
        }));
        assert!(catch_unwind(AssertUnwindSafe(|| poll(panics.as_mut()))).is_err());
        drop(panics);
        drop(profiler);

        let bytes = std::fs::read(&path).unwrap();
        std::fs::remove_file(path).unwrap();
        let report: crate::json::JsonReport = serde_json::from_slice(&bytes).unwrap();
        let alloc = report.functions_alloc.unwrap();
        let count_mode = alloc.profiling_mode == crate::output::ProfilingMode::AllocCount;
        let data = alloc.data;
        assert!(!data.iter().any(|row| row.name == "unpolled block"));
        for (name, calls, bytes, allocations) in [
            ("nested block", 3, "40 B", "5"),
            ("hidden await", 1, "8 B", "1"),
            ("panicking poll", 1, "8 B", "1"),
        ] {
            let row = data.iter().find(|row| row.name == name).unwrap();
            assert_eq!(row.calls, calls);
            assert_eq!(
                row.total,
                if count_mode { allocations } else { bytes },
                "{name}"
            );
        }
    }
}
