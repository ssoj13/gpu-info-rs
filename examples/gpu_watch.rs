//! Samples `gpu_info::stats::query()` for a few seconds so a human can compare the numbers
//! against Task Manager and the vendor's own tool. Not a test: the values depend on what the
//! machine is doing.
fn main() {
    for _ in 0..12 {
        match gpu_info::stats::query() {
            Some(s) => println!(
                "{:>5.1}%  {:>5.2}/{:>5.2} GiB  bus {:>4}  {:>5}  {:>6}  {:>7}  {:>4}  {}",
                s.util_pct.unwrap_or(-1.0),
                s.mem_used_bytes.unwrap_or(0) as f64 / (1 << 30) as f64,
                s.mem_total_bytes.unwrap_or(0) as f64 / (1 << 30) as f64,
                s.mem_bus_pct.map_or("-".into(), |v| format!("{v:.0}%")),
                s.temp_c.map_or("-".into(), |v| format!("{v:.0}C")),
                s.power_w.map_or("-".into(), |v| format!("{v:.0}W")),
                s.clock_core_mhz.map_or("-".into(), |v| format!("{v}MHz")),
                s.fan_pct.map_or("-".into(), |v| format!("{v:.0}%")),
                s.name.as_deref().unwrap_or("?"),
            ),
            None => println!("no backend"),
        }
        std::thread::sleep(std::time::Duration::from_millis(400));
    }
}
