//! memory.rs — simple memory pool tracker (placeholder for arena allocator).

use crate::Result;

pub struct MemoryPool {
    total_mb: usize,
    used_mb:  parking_lot::Mutex<usize>,
}

impl MemoryPool {
    pub fn new(total_mb: usize) -> Result<Self> {
        Ok(Self {
            total_mb,
            used_mb: parking_lot::Mutex::new(0),
        })
    }

    pub fn used_mb(&self)  -> usize { *self.used_mb.lock() }
    pub fn total_mb(&self) -> usize { self.total_mb }
    pub fn free_mb(&self)  -> usize { self.total_mb.saturating_sub(self.used_mb()) }

    pub fn track_alloc(&self, mb: usize) {
        *self.used_mb.lock() += mb;
    }
    pub fn track_free(&self, mb: usize) {
        let mut u = self.used_mb.lock();
        *u = u.saturating_sub(mb);
    }
}
