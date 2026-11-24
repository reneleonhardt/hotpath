//! Run with:
//!   cargo run -p test-alloc --example run_totals --features hotpath-alloc

use std::alloc::{GlobalAlloc, Layout};
use std::hint::black_box;

static TRACKED: hotpath::CountingAllocator = hotpath::CountingAllocator::new();

#[hotpath::measure]
fn work(size: usize) {
    let layout = Layout::from_size_align(size, 8).unwrap();
    // SAFETY: the layout has nonzero size and the pointer is freed with the same layout.
    unsafe {
        let ptr = black_box(TRACKED.alloc(layout));
        assert!(!ptr.is_null());
        TRACKED.dealloc(ptr, layout);
    }
}

#[hotpath::measure]
fn worker() {
    CLEANUP.with(|value| {
        black_box(value);
    });
    work(8);
}

struct Cleanup;
impl Drop for Cleanup {
    fn drop(&mut self) {
        work(16);
    }
}

thread_local! { static CLEANUP: Cleanup = Cleanup; }

fn main() {
    let _guard = hotpath::HotpathGuardBuilder::new("run_totals")
        .before_shutdown(|| work(24))
        .build();
    std::thread::spawn(worker).join().unwrap();
}
