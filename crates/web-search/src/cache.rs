//! cache.rs — In-memory TTL cache for search results.

use crate::router::SearchResponse;
use std::sync::RwLock;
use std::collections::HashMap;
use std::time::{Instant, Duration};

pub struct SearchCache {
    store:   RwLock<HashMap<String, (SearchResponse, Instant)>>,
    ttl:     Duration,
}

impl SearchCache {
    pub fn new(ttl_s: u64) -> Self {
        Self {
            store: RwLock::new(HashMap::new()),
            ttl:   Duration::from_secs(ttl_s),
        }
    }

    pub fn get(&self, key: &str) -> Option<SearchResponse> {
        let store = self.store.read().ok()?;
        let (resp, inserted) = store.get(key)?;
        if inserted.elapsed() < self.ttl {
            Some(resp.clone())
        } else {
            None
        }
    }

    pub fn insert(&self, key: &str, resp: SearchResponse) {
        if let Ok(mut store) = self.store.write() {
            // Evict expired entries periodically
            if store.len() > 1000 {
                let ttl = self.ttl;
                store.retain(|_, (_, t)| t.elapsed() < ttl);
            }
            store.insert(key.to_string(), (resp, Instant::now()));
        }
    }
}

//! ratelimit.rs — Per-domain rate limiting using token buckets.

use std::sync::Mutex;
use std::collections::HashMap;
use std::time::{Instant, Duration};

pub struct RateLimiter {
    buckets: Mutex<HashMap<String, TokenBucket>>,
}

struct TokenBucket {
    tokens:      f64,
    max_tokens:  f64,
    refill_rate: f64,   // tokens per second
    last_refill: Instant,
}

impl TokenBucket {
    fn new(rate_per_sec: f64, burst: f64) -> Self {
        Self { tokens: burst, max_tokens: burst, refill_rate: rate_per_sec, last_refill: Instant::now() }
    }

    fn try_consume(&mut self, n: f64) -> bool {
        let elapsed = self.last_refill.elapsed().as_secs_f64();
        self.tokens = (self.tokens + elapsed * self.refill_rate).min(self.max_tokens);
        self.last_refill = Instant::now();
        if self.tokens >= n { self.tokens -= n; true } else { false }
    }
}

impl RateLimiter {
    pub fn new() -> Self { Self { buckets: Mutex::new(HashMap::new()) } }

    pub fn check(&self, domain: &str) -> bool {
        let mut buckets = self.buckets.lock().unwrap();
        let rates: HashMap<&str, (f64, f64)> = [
            ("export.arxiv.org",                    (3.0, 5.0)),
            ("api.semanticscholar.org",              (5.0, 10.0)),
            ("webbook.nist.gov",                     (2.0, 3.0)),
            ("api.adsabs.harvard.edu",               (5.0, 10.0)),
            ("pubchem.ncbi.nlm.nih.gov",             (5.0, 10.0)),
            ("api.search.brave.com",                 (1.0, 3.0)),
            ("api.duckduckgo.com",                   (2.0, 4.0)),
        ].into();
        let (rate, burst) = rates.get(domain).copied().unwrap_or((2.0, 5.0));
        buckets.entry(domain.to_string())
            .or_insert_with(|| TokenBucket::new(rate, burst))
            .try_consume(1.0)
    }
}
