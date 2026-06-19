//! `wgpu-info` — print this system's GPU capabilities (a wgpu-flavoured `vulkaninfo`).

use std::process::ExitCode;

use clap::Parser;
use gpu_info::{GpuReport, wgpu};

#[derive(Parser)]
#[command(name = "wgpu-info", version, about = "Report GPU capabilities via wgpu")]
struct Args {
    /// Emit the full report as JSON instead of a human-readable table.
    #[arg(long)]
    json: bool,

    /// Restrict enumeration to one backend: vulkan | dx12 | metal | gl | primary | all.
    #[arg(long, default_value = "all")]
    backend: String,

    /// Show only the adapter at this index.
    #[arg(long)]
    adapter: Option<usize>,

    /// Compare the live report against a previously saved JSON report and print the differences.
    #[arg(long, value_name = "FILE")]
    diff: Option<String>,
}

fn parse_backends(name: &str) -> Result<wgpu::Backends, String> {
    Ok(match name.to_ascii_lowercase().as_str() {
        "all" => wgpu::Backends::all(),
        "primary" => wgpu::Backends::PRIMARY,
        "vulkan" | "vk" => wgpu::Backends::VULKAN,
        "dx12" | "d3d12" => wgpu::Backends::DX12,
        "metal" | "mtl" => wgpu::Backends::METAL,
        "gl" | "opengl" | "gles" => wgpu::Backends::GL,
        other => return Err(format!("unknown backend '{other}' (use vulkan|dx12|metal|gl|primary|all)")),
    })
}

fn run() -> Result<ExitCode, Box<dyn std::error::Error>> {
    let args = Args::parse();
    let backends = parse_backends(&args.backend)?;

    let mut report = gpu_info::query_backends(backends);

    if let Some(idx) = args.adapter {
        if idx >= report.adapters.len() {
            return Err(format!(
                "adapter index {idx} out of range ({} adapters found)",
                report.adapters.len()
            )
            .into());
        }
        report.adapters = vec![report.adapters.swap_remove(idx)];
    }

    if let Some(path) = args.diff {
        let baseline: GpuReport = serde_json::from_str(&std::fs::read_to_string(&path)?)?;
        let diffs = baseline.diff(&report);
        if diffs.is_empty() {
            println!("no differences");
            return Ok(ExitCode::SUCCESS);
        }
        println!("{} difference(s) vs {path}:", diffs.len());
        for d in diffs {
            println!("  {d}");
        }
        // Non-zero so the diff can gate CI / regression checks.
        return Ok(ExitCode::FAILURE);
    }

    if args.json {
        println!("{}", serde_json::to_string_pretty(&report)?);
    } else {
        print!("{}", report.to_pretty());
    }
    Ok(ExitCode::SUCCESS)
}

fn main() -> ExitCode {
    match run() {
        Ok(code) => code,
        Err(e) => {
            eprintln!("error: {e}");
            ExitCode::FAILURE
        }
    }
}
