use std::time::Duration;
use std::{f32, f64};

fn generate_linear_naively32(
    start_freq_hz: f64,
    end_freq_hz: f64,
    sampling_rate: f64,
    duration: Duration,
) -> Vec<f32> {
    let start_freq_hz = start_freq_hz as f32;
    let end_freq_hz = end_freq_hz as f32;
    let sampling_rate = sampling_rate as f32;
    let freq_slope_hz = (end_freq_hz - start_freq_hz) / duration.as_secs_f32();
    let total_samples = (sampling_rate * duration.as_secs_f32()) as usize;

    let mut result = vec![0.0; total_samples];
    for (sample, x) in result.iter_mut().enumerate() {
        let time = sample as f32 / sampling_rate;
        let freq = 2.0 * f32::consts::PI * (start_freq_hz + freq_slope_hz * time);
        let phase = freq * time;

        *x = phase.sin() as f32;
    }
    return result;
}

fn generate_linear_naively64(
    start_freq_hz: f64,
    end_freq_hz: f64,
    sampling_rate: f64,
    duration: Duration,
) -> Vec<f32> {
    let freq_slope_hz = (end_freq_hz - start_freq_hz) / duration.as_secs_f64();
    let total_samples = (sampling_rate * duration.as_secs_f64()) as usize;

    let mut result = vec![0.0; total_samples];
    for (sample, x) in result.iter_mut().enumerate() {
        let time = sample as f64 / sampling_rate;
        let freq = 2.0 * f64::consts::PI * (start_freq_hz + freq_slope_hz * time);
        let phase = freq * time;

        *x = phase.sin() as f32;
    }
    return result;
}

fn generate_linear_by_rectangle32(
    start_freq_hz: f64,
    end_freq_hz: f64,
    sampling_rate: f64,
    duration: Duration,
    use_rem: bool,
) -> Vec<f32> {
    let start_freq_hz = start_freq_hz as f32;
    let end_freq_hz = end_freq_hz as f32;
    let sampling_rate = sampling_rate as f32;
    let freq_slope_hz = (end_freq_hz - start_freq_hz) / duration.as_secs_f32();
    let total_samples = (sampling_rate * duration.as_secs_f32()) as usize;

    let mut accumulator: f32 = 0.0;
    let mut result = vec![0.0; total_samples];
    for (sample, x) in result.iter_mut().enumerate() {
        *x = accumulator.sin() as f32;

        let time = sample as f32 / sampling_rate;
        let freq = 2.0 * f32::consts::PI * (start_freq_hz + freq_slope_hz * time);

        let phase_inc = freq / sampling_rate;
        accumulator = accumulator + phase_inc;
        if use_rem {
            accumulator = accumulator.rem_euclid(f32::consts::TAU);
        }
    }
    return result;
}

fn generate_linear_by_rectangle64(
    start_freq_hz: f64,
    end_freq_hz: f64,
    sampling_rate: f64,
    duration: Duration,
    use_rem: bool,
) -> Vec<f32> {
    let freq_slope_hz = (end_freq_hz - start_freq_hz) / duration.as_secs_f64();
    let total_samples = (sampling_rate * duration.as_secs_f64()) as usize;

    let mut accumulator: f64 = 0.0;
    let mut result = vec![0.0; total_samples];
    for (sample, x) in result.iter_mut().enumerate() {
        *x = accumulator.sin() as f32;

        let time = sample as f64 / sampling_rate;
        let freq = 2.0 * f64::consts::PI * (start_freq_hz + freq_slope_hz * time);

        let phase_inc = freq / sampling_rate;
        accumulator = accumulator + phase_inc;
        if use_rem {
            accumulator = accumulator.rem_euclid(f64::consts::TAU);
        }
    }
    return result;
}

fn generate_linear_by_trapezoid32(
    start_freq_hz: f64,
    end_freq_hz: f64,
    sampling_rate: f64,
    duration: Duration,
    use_rem: bool,
) -> Vec<f32> {
    let start_freq_hz = start_freq_hz as f32;
    let end_freq_hz = end_freq_hz as f32;
    let sampling_rate = sampling_rate as f32;
    let freq_slope_hz = (end_freq_hz - start_freq_hz) / duration.as_secs_f32();
    let total_samples = (sampling_rate * duration.as_secs_f32()) as usize;

    let mut accumulator: f32 = 0.0;
    let mut result = vec![0.0; total_samples];
    for (sample, x) in result.iter_mut().enumerate() {
        *x = accumulator.sin() as f32;

        let time_current = sample as f32 / sampling_rate;
        let freq_current = 2.0 * f32::consts::PI * (start_freq_hz + freq_slope_hz * time_current);
        let time_next = (sample + 1) as f32 / sampling_rate;
        let freq_next = 2.0 * f32::consts::PI * (start_freq_hz + freq_slope_hz * time_next);
        let freq_average = (freq_current + freq_next) / 2.0;

        let phase_inc = freq_average / sampling_rate;
        accumulator = accumulator + phase_inc;
        if use_rem {
            accumulator = accumulator.rem_euclid(f32::consts::TAU);
        }
    }
    return result;
}

fn generate_linear_by_trapezoid64(
    start_freq_hz: f64,
    end_freq_hz: f64,
    sampling_rate: f64,
    duration: Duration,
    use_rem: bool,
) -> Vec<f32> {
    let freq_slope_hz = (end_freq_hz - start_freq_hz) / duration.as_secs_f64();
    let total_samples = (sampling_rate * duration.as_secs_f64()) as usize;

    let mut accumulator: f64 = 0.0;
    let mut result = vec![0.0; total_samples];
    for (sample, x) in result.iter_mut().enumerate() {
        *x = accumulator.sin() as f32;

        let time_current = sample as f64 / sampling_rate;
        let freq_current = 2.0 * f64::consts::PI * (start_freq_hz + freq_slope_hz * time_current);
        let time_next = (sample + 1) as f64 / sampling_rate;
        let freq_next = 2.0 * f64::consts::PI * (start_freq_hz + freq_slope_hz * time_next);
        let freq_average = (freq_current + freq_next) / 2.0;

        let phase_inc = freq_average / sampling_rate;
        accumulator = accumulator + phase_inc;
        if use_rem {
            accumulator = accumulator.rem_euclid(f64::consts::TAU);
        }
    }
    return result;
}

fn generate_linear_by_trapezoid_mixed(
    start_freq_hz: f64,
    end_freq_hz: f64,
    sampling_rate: f64,
    duration: Duration,
    use_rem: bool,
) -> Vec<f32> {
    let start_freq_hz = start_freq_hz as f32;
    let end_freq_hz = end_freq_hz as f32;
    let freq_slope_hz = (end_freq_hz - start_freq_hz) / duration.as_secs_f32();
    let total_samples = (sampling_rate * duration.as_secs_f64()) as usize;

    let mut accumulator: f64 = 0.0;
    let mut result = vec![0.0; total_samples];
    for (sample, x) in result.iter_mut().enumerate() {
        *x = accumulator.sin() as f32;

        let time_current = sample as f32 / sampling_rate as f32;
        let freq_current = 2.0 * f32::consts::PI * (start_freq_hz + freq_slope_hz * time_current);
        let time_next = (sample + 1) as f32 / sampling_rate as f32;
        let freq_next = 2.0 * f32::consts::PI * (start_freq_hz + freq_slope_hz * time_next);
        let freq_average = (freq_current + freq_next) / 2.0;

        let phase_inc = freq_average as f64 / sampling_rate;
        accumulator = accumulator + phase_inc;
        if use_rem {
            accumulator = accumulator.rem_euclid(f64::consts::TAU);
        }
    }
    return result;
}

fn generate_linear_analytically(
    start_freq_hz: f64,
    end_freq_hz: f64,
    sampling_rate: f64,
    duration: Duration,
) -> Vec<f32> {
    let freq_slope_hz = (end_freq_hz - start_freq_hz) / duration.as_secs_f64();
    let total_samples = (sampling_rate * duration.as_secs_f64()) as usize;

    let mut result = vec![0.0; total_samples];
    for (sample, x) in result.iter_mut().enumerate() {
        let time = sample as f64 / sampling_rate;
        let phase =
            2.0 * f64::consts::PI * (start_freq_hz * time + 0.5 * freq_slope_hz * time * time);

        *x = phase.sin() as f32;
    }
    return result;
}

fn generate_quadratic_by_rectangle32(
    start_freq_hz: f64,
    sampling_rate: f64,
    duration: Duration,
    use_rem: bool,
) -> Vec<f32> {
    let start_freq_hz = start_freq_hz as f32;
    let sampling_rate = sampling_rate as f32;
    let total_samples = (sampling_rate * duration.as_secs_f32()) as usize;

    let mut accumulator: f32 = 0.0;
    let mut result = vec![0.0; total_samples];
    //println!("time, freq, phase_inc, accumulator");
    for (sample, x) in result.iter_mut().enumerate() {
        *x = accumulator.sin() as f32;

        let time = sample as f32 / sampling_rate;
        let freq = 2.0 * f32::consts::PI * (start_freq_hz + time * time);

        let phase_inc = freq / sampling_rate;
        //println!("{}, {}, {}, {}", time, freq, phase_inc, accumulator);
        accumulator = accumulator + phase_inc;
        if use_rem {
            accumulator = accumulator.rem_euclid(f32::consts::TAU);
        }
    }
    return result;
}

fn generate_quadratic_by_trapezoid32(
    start_freq_hz: f64,
    sampling_rate: f64,
    duration: Duration,
    use_rem: bool,
) -> Vec<f32> {
    let start_freq_hz = start_freq_hz as f32;
    let sampling_rate = sampling_rate as f32;
    let total_samples = (sampling_rate * duration.as_secs_f32()) as usize;

    let mut accumulator: f32 = 0.0;
    let mut result = vec![0.0; total_samples];
    for (sample, x) in result.iter_mut().enumerate() {
        *x = accumulator.sin() as f32;

        let time_current = sample as f32 / sampling_rate;
        let freq_current = 2.0 * f32::consts::PI * (start_freq_hz + time_current * time_current);
        let time_next = (sample + 1) as f32 / sampling_rate;
        let freq_next = 2.0 * f32::consts::PI * (start_freq_hz + time_next * time_next);
        let freq_average = (freq_current + freq_next) / 2.0;

        let phase_inc = freq_average / sampling_rate;
        accumulator = accumulator + phase_inc;
        if use_rem {
            accumulator = accumulator.rem_euclid(f32::consts::TAU);
        }
    }
    return result;
}

fn generate_quadratic_analytically(
    start_freq_hz: f64,
    sampling_rate: f64,
    duration: Duration,
) -> Vec<f32> {
    let total_samples = (sampling_rate * duration.as_secs_f64()) as usize;

    let mut result = vec![0.0; total_samples];
    let mut last_phase = 0.0;
    //println!("time, freq, phase_inc, phase");
    for (sample, x) in result.iter_mut().enumerate() {
        let time = sample as f64 / sampling_rate;
        let phase = 2.0 * f64::consts::PI * (start_freq_hz * time + time * time * time / 3.0);
        //println!("{}, {}, {}, {}", time, 2.0 * f64::consts::PI * (start_freq_hz + time * time), phase - last_phase, phase.rem_euclid(f64::consts::TAU));
        last_phase = phase;
        *x = phase.sin() as f32;
    }
    _ = last_phase;
    return result;
}

fn generate_piecewise_cubic_by_rectangle32(
    start_freq_hz: f64,
    sampling_rate: f64,
    duration: Duration,
    use_rem: bool,
) -> Vec<f32> {
    let start_freq_hz = start_freq_hz as f32;
    let sampling_rate = sampling_rate as f32;
    let total_samples = (sampling_rate * duration.as_secs_f32()) as usize;

    let mut accumulator: f32 = 0.0;
    let mut result = vec![0.0; total_samples];
    //println!("time, freq, phase_inc, accumulator");
    for (sample, x) in result.iter_mut().enumerate() {
        *x = accumulator.sin() as f32;

        let time = (sample as f32 / sampling_rate).rem_euclid(10.0);
        let freq = 2.0 * f32::consts::PI * (start_freq_hz + time * time * time);

        let phase_inc = freq / sampling_rate;
        //println!("{}, {}, {}, {}", time, freq, phase_inc, accumulator);
        accumulator = accumulator + phase_inc;
        if use_rem {
            accumulator = accumulator.rem_euclid(f32::consts::TAU);
        }
    }
    return result;
}

fn generate_piecewise_cubic_analytically(
    start_freq_hz: f64,
    sampling_rate: f64,
    duration: Duration,
) -> Vec<f32> {
    let total_samples = (sampling_rate * duration.as_secs_f64()) as usize;

    let mut result = vec![0.0; total_samples];
    let mut last_phase = 0.0;
    //println!("time, freq, phase_inc, phase");
    for (sample, x) in result.iter_mut().enumerate() {
        let time = (sample as f64 / sampling_rate).rem_euclid(10.0);
        let phase =
            2.0 * f64::consts::PI * (start_freq_hz * time + time * time * time * time / 4.0);
        //println!("{}, {}, {}, {}", time, 2.0 * f64::consts::PI * (start_freq_hz + time * time * time), phase - last_phase, phase.rem_euclid(f64::consts::TAU));
        last_phase = phase;
        *x = phase.sin() as f32;
    }
    _ = last_phase;
    return result;
}

fn compute_diff_and_write_wav(
    spec: hound::WavSpec,
    name: String,
    buf: &Vec<f32>,
    correct: &Vec<f32>,
) {
    let mut writer = hound::WavWriter::create(format!("{}.wav", name), spec).unwrap();
    let mut diff = 0.0;
    for (i, x) in buf.iter().enumerate() {
        diff += (x - correct[i]).abs() as f64;
        writer.write_sample(*x).unwrap();
    }
    writer.finalize().unwrap();
    println!("Average diff for {}: {}", name, diff / buf.len() as f64);
}

pub fn main() {
    let start_freq_hz = 0.0;
    let end_freq_hz = 10000.0;

    let sampling_rate = 44100.0; // Hz
    let duration = Duration::from_secs(10);

    let spec = hound::WavSpec {
        channels: 1,
        sample_rate: sampling_rate as u32,
        bits_per_sample: 32,
        sample_format: hound::SampleFormat::Float,
    };

    let linear_correct =
        generate_linear_analytically(start_freq_hz, end_freq_hz, sampling_rate, duration);

    compute_diff_and_write_wav(
        spec,
        "naively32".to_string(),
        &&generate_linear_naively32(start_freq_hz, end_freq_hz, sampling_rate, duration),
        &linear_correct,
    );

    compute_diff_and_write_wav(
        spec,
        "naively64".to_string(),
        &&generate_linear_naively64(start_freq_hz, end_freq_hz, sampling_rate, duration),
        &linear_correct,
    );

    compute_diff_and_write_wav(
        spec,
        "rectangle32-no-rem".to_string(),
        &generate_linear_by_rectangle32(start_freq_hz, end_freq_hz, sampling_rate, duration, false),
        &linear_correct,
    );

    compute_diff_and_write_wav(
        spec,
        "rectangle32".to_string(),
        &generate_linear_by_rectangle32(start_freq_hz, end_freq_hz, sampling_rate, duration, true),
        &linear_correct,
    );

    compute_diff_and_write_wav(
        spec,
        "rectangle64-no-rem".to_string(),
        &generate_linear_by_rectangle64(start_freq_hz, end_freq_hz, sampling_rate, duration, false),
        &linear_correct,
    );

    compute_diff_and_write_wav(
        spec,
        "rectangle64".to_string(),
        &generate_linear_by_rectangle64(start_freq_hz, end_freq_hz, sampling_rate, duration, true),
        &linear_correct,
    );

    compute_diff_and_write_wav(
        spec,
        "trapezoid32-no-rem".to_string(),
        &generate_linear_by_trapezoid32(start_freq_hz, end_freq_hz, sampling_rate, duration, false),
        &linear_correct,
    );

    compute_diff_and_write_wav(
        spec,
        "trapezoid32".to_string(),
        &generate_linear_by_trapezoid32(start_freq_hz, end_freq_hz, sampling_rate, duration, true),
        &linear_correct,
    );

    compute_diff_and_write_wav(
        spec,
        "trapezoid64-no-rem".to_string(),
        &generate_linear_by_trapezoid64(start_freq_hz, end_freq_hz, sampling_rate, duration, false),
        &linear_correct,
    );

    compute_diff_and_write_wav(
        spec,
        "trapezoid64".to_string(),
        &generate_linear_by_trapezoid64(start_freq_hz, end_freq_hz, sampling_rate, duration, true),
        &linear_correct,
    );

    compute_diff_and_write_wav(
        spec,
        "trapezoid-mixed".to_string(),
        &generate_linear_by_trapezoid_mixed(
            start_freq_hz,
            end_freq_hz,
            sampling_rate,
            duration,
            true,
        ),
        &linear_correct,
    );

    compute_diff_and_write_wav(
        spec,
        "linear-correct".to_string(),
        &linear_correct,
        &linear_correct,
    );

    let quadratic_correct = generate_quadratic_analytically(start_freq_hz, sampling_rate, duration);

    compute_diff_and_write_wav(
        spec,
        "quadratic-rectangle32".to_string(),
        &generate_quadratic_by_rectangle32(start_freq_hz, sampling_rate, duration, true),
        &quadratic_correct,
    );

    compute_diff_and_write_wav(
        spec,
        "quadratic-trapezoid32".to_string(),
        &generate_quadratic_by_trapezoid32(start_freq_hz, sampling_rate, duration, true),
        &quadratic_correct,
    );

    compute_diff_and_write_wav(
        spec,
        "quadratic-correct".to_string(),
        &quadratic_correct,
        &quadratic_correct,
    );

    let piecewise_cubic_correct =
        generate_piecewise_cubic_analytically(start_freq_hz, sampling_rate, duration);

    compute_diff_and_write_wav(
        spec,
        "piecewise-cubic-rectangle32".to_string(),
        &generate_piecewise_cubic_by_rectangle32(start_freq_hz, sampling_rate, duration, true),
        &piecewise_cubic_correct,
    );

    compute_diff_and_write_wav(
        spec,
        "piecewise-cubic-correct".to_string(),
        &piecewise_cubic_correct,
        &piecewise_cubic_correct,
    );
}
