// Original source: https://github.com/fornwall/allocation-counter
//
// Licensed under either of:
// - Apache License, Version 2.0.
// - MIT/X Consortium License
//
// Modifications:
// - Adjusted to work with hotpath module system
// - Split into feature-specific dispatching allocator

use std::alloc::{GlobalAlloc, Layout, System};

/// Shared global allocator that tracks allocations when the `hotpath-alloc`
/// feature is enabled and forwards to the inner allocator unchanged otherwise.
pub struct CountingAllocator<A = System>(A);

impl CountingAllocator<System> {
    pub const fn new() -> Self {
        Self(System)
    }
}

impl<A> CountingAllocator<A> {
    pub const fn with(inner: A) -> Self {
        Self(inner)
    }
}

impl<A: Default> Default for CountingAllocator<A> {
    fn default() -> Self {
        Self(A::default())
    }
}

// SAFETY: pure pass-through to the inner allocator `A` - every pointer
// returned by `alloc`/`alloc_zeroed`/`realloc` comes from the corresponding
// method of `A`, and `dealloc`/`realloc` forward the caller's
// `ptr`/`layout`/`new_size` unchanged, so `A`'s GlobalAlloc guarantees carry
// over. The tracking hooks only update thread-local counters and never touch
// the allocation itself.
unsafe impl<A> GlobalAlloc for CountingAllocator<A>
where
    A: GlobalAlloc,
{
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        // SAFETY: the caller upholds GlobalAlloc's contract for `layout`.
        let ptr = unsafe { self.0.alloc(layout) };

        #[cfg(feature = "hotpath-alloc")]
        if !ptr.is_null() {
            crate::lib_on::functions::alloc::core::track_alloc(layout.size());
        }
        ptr
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        #[cfg(feature = "hotpath-alloc")]
        crate::lib_on::functions::alloc::core::track_dealloc(layout.size());

        // SAFETY: caller guarantees `ptr` was allocated by this allocator
        // with `layout`, which means it came from `A::alloc`; both are
        // forwarded unchanged.
        unsafe {
            self.0.dealloc(ptr, layout);
        }
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        // SAFETY: the caller upholds GlobalAlloc's contract for `layout`.
        let ptr = unsafe { self.0.alloc_zeroed(layout) };

        #[cfg(feature = "hotpath-alloc")]
        if !ptr.is_null() {
            crate::lib_on::functions::alloc::core::track_alloc(layout.size());
        }
        ptr
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        // SAFETY: caller guarantees `ptr` was allocated by this allocator
        // with `layout` and that `new_size` is valid per GlobalAlloc's
        // contract; all three are forwarded unchanged.
        let new_ptr = unsafe { self.0.realloc(ptr, layout, new_size) };

        // Track only on success: a failed realloc leaves the old block alive,
        // so recording a dealloc for it would over-count frees.
        #[cfg(feature = "hotpath-alloc")]
        if !new_ptr.is_null() {
            crate::lib_on::functions::alloc::core::track_dealloc(layout.size());
            crate::lib_on::functions::alloc::core::track_alloc(new_size);
        }
        new_ptr
    }
}

#[cfg(all(test, feature = "hotpath-alloc"))]
mod tests {
    use std::alloc::{GlobalAlloc, Layout, System};

    use crate::lib_on::functions::alloc::guard::{pop_alloc_stack, push_alloc_stack};
    use crate::CountingAllocator;

    struct Limited;

    unsafe impl GlobalAlloc for Limited {
        unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
            if layout.size() == 8 {
                unsafe { System.alloc(layout) }
            } else {
                std::ptr::null_mut()
            }
        }

        unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
            unsafe { System.dealloc(ptr, layout) }
        }
    }

    #[test]
    fn counts_only_successful_allocations() {
        let alloc = CountingAllocator::with(Limited);
        push_alloc_stack();
        unsafe {
            for (size, zeroed) in [(8, false), (8, true), (32, false), (64, true)] {
                let layout = Layout::from_size_align(size, 1).unwrap();
                let ptr = if zeroed {
                    alloc.alloc_zeroed(layout)
                } else {
                    alloc.alloc(layout)
                };
                assert_eq!(ptr.is_null(), size != 8);
                if !ptr.is_null() {
                    if zeroed {
                        assert_eq!(ptr.read(), 0);
                    }
                    ptr.write(0xa5);
                    assert!(alloc.realloc(ptr, layout, 32).is_null());
                    assert_eq!(ptr.read(), 0xa5);

                    let new_ptr = alloc.realloc(ptr, layout, 8);
                    assert!(!new_ptr.is_null());
                    assert_eq!(new_ptr.read(), 0xa5);
                    alloc.dealloc(new_ptr, layout);
                }
            }
        }
        assert_eq!(pop_alloc_stack(), (32, 4));
    }
}
