use std::ops::Add;
use std::time::Duration;

#[cfg(test)]
use mock_instant::thread_local::Instant;
#[cfg(not(test))]
use std::time::Instant;

/* A simple time series metric that stores values for a finite period of time and with a
 * fixed set of time buckets. */
pub struct Metric<T>
where
    T: Add + Copy + Default,
{
    duration: Duration,
    data: Vec<T>,
    start: Instant,
    last_write: Instant,
}

impl<T> Metric<T>
where
    T: Add<Output = T> + Copy + Default,
{
    pub fn new(dur: Duration, buckets: usize) -> Self {
        let data = vec![T::default(); buckets];
        Metric {
            duration: dur,
            data,
            start: Instant::now(),
            last_write: Instant::now() - dur,
        }
    }

    fn bucket_from_time(&self, time: Instant) -> usize {
        let elapsed = (time - self.start).as_secs_f64();
        let bucket_duration = self.duration.as_secs_f64() / self.data.len() as f64;
        (elapsed / bucket_duration) as usize % self.data.len()
    }

    fn current_bucket(&mut self, for_write: bool) -> usize {
        let now = Instant::now();
        let current_bucket = self.bucket_from_time(now);
        if now - self.last_write >= self.duration {
            // Reset all buckets
            for i in 0..self.data.len() {
                self.data[i] = T::default();
            }
        } else if current_bucket < self.bucket_from_time(self.last_write) {
            for i in 0..current_bucket {
                self.data[i] = T::default();
            }
        }
        if current_bucket != self.bucket_from_time(self.last_write) {
            self.data[current_bucket] = T::default();
        }
        if for_write {
            self.last_write = now;
        }
        return current_bucket;
    }
    /*
    pub fn add(&mut self, value: T) {
        let bucket = self.current_bucket();
        self.data[bucket] = self.data[bucket] + value;
    }
    */
    pub fn set(&mut self, value: T) {
        let bucket = self.current_bucket(true);
        self.data[bucket] = value;
    }

    // Returns an iterator over the data in the metric, starting from the oldest bucket.
    pub fn iter(&mut self) -> impl Iterator<Item = T> + use<'_, T> {
        let current_bucket = self.current_bucket(false);
        let len = self.data.len();
        (1..len + 1).map(move |i| self.data[(current_bucket + i) % len])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mock_instant::thread_local::MockClock;

    #[test]
    fn test_metric() {
        MockClock::set_time(Duration::from_secs(60));
        let mut metric = Metric::new(Duration::from_secs(5), 5);
        metric.set(1);
        assert_eq!(metric.iter().collect::<Vec<_>>(), vec![0, 0, 0, 0, 1]);

        metric.set(2);
        assert_eq!(metric.iter().collect::<Vec<_>>(), vec![0, 0, 0, 0, 2]);

        MockClock::advance(Duration::from_secs(1));
        metric.set(3);
        MockClock::advance(Duration::from_secs(1));
        metric.set(4);
        assert_eq!(metric.iter().collect::<Vec<_>>(), vec![0, 0, 2, 3, 4]);

        MockClock::advance(Duration::from_secs(1));
        assert_eq!(metric.iter().collect::<Vec<_>>(), vec![0, 2, 3, 4, 0]);

        metric.set(5);
        MockClock::advance(Duration::from_secs(1));
        metric.set(6);
        assert_eq!(metric.iter().collect::<Vec<_>>(), vec![2, 3, 4, 5, 6]);

        MockClock::advance(Duration::from_secs(1));
        assert_eq!(metric.iter().collect::<Vec<_>>(), vec![3, 4, 5, 6, 0]);

        MockClock::advance(Duration::from_secs(3));
        metric.set(7);
        assert_eq!(metric.iter().collect::<Vec<_>>(), vec![6, 0, 0, 0, 7]);

        MockClock::advance(Duration::from_secs(9));
        metric.set(8);
        assert_eq!(metric.iter().collect::<Vec<_>>(), vec![0, 0, 0, 0, 8]);
    }
}
