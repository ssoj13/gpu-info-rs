//! Samples `gpu_info::stats::query()` for a few seconds so a human can compare the numbers
//! against Task Manager. Not a test: the values depend on what the machine is doing.
fn main() {
    for _ in 0..20 {
        match gpu_info::stats::query() {
            Some(s) => println!(
                "{:>6.1}%  {:>6.2} / {:>6.2} GiB  {}",
                s.util_pct.unwrap_or(-1.0),
                s.mem_used_bytes.unwrap_or(0) as f64 / (1 << 30) as f64,
                s.mem_total_bytes.unwrap_or(0) as f64 / (1 << 30) as f64,
                s.name.as_deref().unwrap_or("?"),
            ),
            None => println!("no backend"),
        }
        std::thread::sleep(std::time::Duration::from_millis(300));
    }
}
