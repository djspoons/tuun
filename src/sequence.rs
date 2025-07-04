use std::f32::consts::PI;
use std::sync::mpsc::{Receiver, Sender};

extern crate sdl2;
use sdl2::audio::AudioCallback;

use crate::parser::{BuiltInFn, Expr};
use crate::Command;

trait Generator: Send {
    fn generate(&self, offset: usize, out: &mut [f32]);
    fn samples(&self) -> Option<usize> {
        None
    }
    fn next_offset(&self) -> Option<usize> {
        None
    }
}

// Sinusoidal wave generator with the given frequency
struct SineWave {
    sample_frequency: i32,
    tone_frequency: f32,
}
impl Generator for SineWave {
    fn generate(&self, offset: usize, out: &mut [f32]) {
        for (i, f) in out.iter_mut().enumerate() {
            let t_secs = (i + offset) as f32 / self.sample_frequency as f32;
            *f = (2.0 * PI * self.tone_frequency * t_secs).sin();
        }
    }
}

struct Amplify {
    level: f32,
    generator: Box<dyn Generator>,
}
impl Generator for Amplify {
    fn generate(&self, offset: usize, out: &mut [f32]) {
        self.generator.generate(offset, out);
        for x in out.iter_mut() {
            *x *= self.level;
        }
    }
    fn samples(&self) -> Option<usize> {
        self.generator.samples()
    }
    fn next_offset(&self) -> Option<usize> {
        self.generator.next_offset()
    }
}

/*
 * LinearRamp is a time-dependent filter that modifies the given generator, starting after the
 * next_offset of that generator. The output is multiplied by a linear ramp that starts at
 * initial_level and ends at final_level. If the generator has no next_offset then the ramp
 * starts at the beginning of the output.
 */
struct LinearRamp {
    initial_level: f32,
    length: usize, // length in samples
    final_level: f32,
    generator: Box<dyn Generator>,
}
impl Generator for LinearRamp {
    fn generate(&self, offset: usize, out: &mut [f32]) {
        self.generator.generate(offset, out);
        let inner_offset = self.generator.next_offset().unwrap_or(0);
        let level_diff = (self.final_level - self.initial_level).abs();
        let mut offset = offset;
        for x in out.iter_mut() {
            if offset >= inner_offset && offset < inner_offset + self.length {
                let mut progress: f32 = (offset - inner_offset) as f32 / self.length as f32;
                if self.final_level < self.initial_level {
                    progress = 1.0 - progress;
                }
                *x *= progress * level_diff;
            }
            offset += 1;
        }
    }
    fn samples(&self) -> Option<usize> {
        self.generator.samples()
    }
    fn next_offset(&self) -> Option<usize> {
        match self.generator.next_offset() {
            None => Some(self.length),
            Some(offset) => Some(offset + self.length),
        }
    }
}

/*
 * Sustain is a time-dependent filter that amplifies the portion of a waveform starting at the
 * next_offset of the underlying generator and continuing for the given length.
 */
struct Sustain {
    level: f32,
    length: usize, // length in samples
    generator: Box<dyn Generator>,
}
impl Generator for Sustain {
    fn generate(&self, offset: usize, out: &mut [f32]) {
        self.generator.generate(offset, out);
        let inner_offset = self.generator.next_offset().unwrap_or(0);
        for (i, x) in out.iter_mut().enumerate() {
            if i + offset >= inner_offset && i + offset < inner_offset + self.length {
                *x *= self.level;
            }
        }
    }
    fn samples(&self) -> Option<usize> {
        self.generator.samples()
    }
    fn next_offset(&self) -> Option<usize> {
        match self.generator.next_offset() {
            None => Some(self.length),
            Some(offset) => Some(offset + self.length),
        }
    }
}

/*
 * Seq sets the next_offset to the given value (ignoring next_offset of the underlying generator).
*/
struct Seq {
    generator: Box<dyn Generator>,
    next_offset: usize,
}
impl Generator for Seq {
    fn generate(&self, offset: usize, out: &mut [f32]) {
        self.generator.generate(offset, out);
    }
    fn samples(&self) -> Option<usize> {
        self.generator.samples()
    }
    fn next_offset(&self) -> Option<usize> {
        Some(self.next_offset)
    }
}

struct Fin {
    generator: Box<dyn Generator>,
    samples: usize,
}
impl Generator for Fin {
    fn generate(&self, offset: usize, out: &mut [f32]) {
        self.generator.generate(offset, out);
    }
    fn samples(&self) -> Option<usize> {
        Some(self.samples)
    }
    fn next_offset(&self) -> Option<usize> {
        self.generator.next_offset()
    }
}

struct Chord {
    generators: Vec<Box<dyn Generator>>,
    samples: Option<usize>,
    next_offset: Option<usize>,
}
impl Generator for Chord {
    fn generate(&self, offset: usize, out: &mut [f32]) {
        for generator in &self.generators {
            match generator.samples() {
                None => {
                    let mut tmp = vec![0.0; out.len()];
                    generator.generate(offset, &mut tmp);
                    for (i, x) in tmp.iter().enumerate() {
                        out[i] += x;
                    }
                }
                Some(samples) => {
                    if offset < samples {
                        let len = samples - offset;
                        let mut tmp = vec![0.0; out.len().min(len)];
                        generator.generate(offset, &mut tmp);
                        for (i, x) in tmp.iter().enumerate() {
                            out[i] += x;
                        }
                    }
                }
            }
        }
    }
    fn samples(&self) -> Option<usize> {
        self.samples
    }
    fn next_offset(&self) -> Option<usize> {
        self.next_offset
    }
}

/*
 * Play the generators in sequence, starting each after next_offset of the previous generator.
 * If a given generator doesn't provide next_offset then no subsequent generators will be played.
 */
struct Sequence {
    generators: Vec<Box<dyn Generator>>,
    samples: Option<usize>,
    next_offset: Option<usize>,
}
impl<'a> Generator for Sequence {
    fn generate(&self, offset: usize, mut out: &mut [f32]) {
        // local_offset is the offset within the current generator, starting with the first
        let mut local_offset = offset;
        for generator in &self.generators {
            match generator.samples() {
                None => {
                    generator.generate(offset, out);
                }
                Some(samples) => {
                    if local_offset < samples {
                        let len = samples - local_offset;
                        // We could optimize the case local_offset + next_offset is greater than
                        // out.len() in which case we don't need to allocate a tmp.
                        let mut tmp = vec![0.0; out.len().min(len)];
                        generator.generate(local_offset, &mut tmp);
                        for (i, x) in tmp.iter().enumerate() {
                            out[i] += x;
                        }
                    }
                }
            }
            match generator.next_offset() {
                None => {
                    return;
                }
                Some(next_offset) => {
                    if local_offset + out.len() > next_offset {
                        if local_offset < next_offset {
                            out = &mut out[(next_offset - local_offset)..];
                            local_offset = 0;
                        } else {
                            local_offset -= next_offset;
                        }
                    } else {
                        return;
                    }
                }
            }
        }
    }
    fn samples(&self) -> Option<usize> {
        self.samples
    }
    fn next_offset(&self) -> Option<usize> {
        self.next_offset
    }
}

fn from_expr(sample_frequency: i32, expr: Expr) -> Option<Box<dyn Generator>> {
    use Expr::{Application, BuiltIn, Float, Tuple};
    println!("from_expr called with expr {:?}", expr);
    match expr {
        Application { function, argument } => {
            match (*function, *argument) {
                (BuiltIn(BuiltInFn::SineWave), Float(tone_frequency)) => {
                    return Some(Box::new(SineWave {
                        sample_frequency,
                        tone_frequency,
                    }));
                }
                (BuiltIn(BuiltInFn::Amplify), Tuple(arguments)) if arguments.len() == 2 => {
                    if let (Float(level), waveform) = (&arguments[0], &arguments[1]) {
                        let generator = from_expr(sample_frequency, waveform.clone())?;
                        return Some(Box::new(Amplify {
                            level: *level,
                            generator,
                        }));
                    }
                }

                (BuiltIn(BuiltInFn::Seq), Tuple(arguments)) if arguments.len() == 2 => {
                    if let (Float(duration), waveform) = (&arguments[0], &arguments[1]) {
                        let generator = from_expr(sample_frequency, waveform.clone())?;
                        return Some(Box::new(Seq {
                            generator,
                            next_offset: *duration as usize * sample_frequency as usize,
                        }));
                    }
                }
                (BuiltIn(BuiltInFn::Fin), Tuple(arguments)) if arguments.len() == 2 => {
                    if let (Float(duration), waveform) = (&arguments[0], &arguments[1]) {
                        let generator = from_expr(sample_frequency, waveform.clone())?;
                        return Some(Box::new(Fin {
                            generator,
                            samples: *duration as usize * sample_frequency as usize,
                        }));
                    }
                }

                (BuiltIn(BuiltInFn::LinearRamp), Tuple(arguments)) if arguments.len() == 4 => {
                    if let (Float(initial_level), Float(duration), Float(final_level), waveform) =
                        (&arguments[0], &arguments[1], &arguments[2], &arguments[3])
                    {
                        let generator = from_expr(sample_frequency, waveform.clone())?;
                        return Some(Box::new(LinearRamp {
                            initial_level: *initial_level,
                            length: (*duration * sample_frequency as f32) as usize,
                            final_level: *final_level,
                            generator,
                        }));
                    }
                }
                (BuiltIn(BuiltInFn::Sustain), Tuple(arguments)) if arguments.len() == 3 => {
                    if let (Float(level), Float(duration), waveform) =
                        (&arguments[0], &arguments[1], &arguments[2])
                    {
                        let generator = from_expr(sample_frequency, waveform.clone())?;
                        return Some(Box::new(Sustain {
                            level: *level,
                            length: (*duration * sample_frequency as f32) as usize,
                            generator,
                        }));
                    }
                }
                _ => {
                    return None;
                }
            }
            return None;
        }
        Expr::Chord(exprs) => {
            let mut generators = Vec::new();
            let mut samples = Some(0usize);
            let mut next_offset = Some(0usize);
            for expr in exprs {
                let generator = from_expr(sample_frequency, expr)?;
                match (samples, generator.samples()) {
                    (Some(s), Some(t)) => {
                        samples = Some(s.max(t));
                    }
                    (_, _) => {
                        samples = None;
                    }
                }
                match (next_offset, generator.next_offset()) {
                    (Some(s), Some(t)) => {
                        next_offset = Some(s.max(t));
                    }
                    (_, _) => {
                        next_offset = None;
                    }
                }
                generators.push(generator);
            }
            return Some(Box::new(Chord {
                generators,
                samples: samples,
                next_offset: next_offset,
            }));
        }
        Expr::Sequence(exprs) => {
            let mut generators = Vec::new();
            let mut samples = Some(0usize);
            let mut next_offset = Some(0usize);
            for expr in exprs {
                let generator = from_expr(sample_frequency, expr)?;
                let new_samples = generator.samples();
                let new_next_offset = generator.next_offset();
                generators.push(generator);
                match (samples, new_samples, next_offset) {
                    (Some(old_samples), Some(s), Some(old_next_offset)) => {
                        samples = Some(old_samples.max(old_next_offset + s));
                    }
                    (_, None, _) => {
                        samples = None;
                    }
                    (None, _, _) => {} // Samples is already None
                    (_, _, None) => {
                        panic!("Unexpected case in sequence generator construction");
                    }
                }
                match (next_offset, new_next_offset) {
                    (Some(old_next_offset), Some(t)) => {
                        next_offset = Some(old_next_offset + t);
                    }
                    (_, None) => {
                        return Some(Box::new(Sequence {
                            generators,
                            samples,
                            next_offset: None,
                        }));
                    }
                    (None, _) => {
                        panic!("Unexpected case in sequence generator construction");
                    }
                }
            }
            return Some(Box::new(Sequence {
                generators,
                samples,
                next_offset,
            }));
        }
        _ => {
            println!("Unsupported expression in from_expr: {:?}", expr);
            None
        }
    }
}

pub struct Tracker {
    sample_frequency: i32,
    _beats_per_minute: i32,
    command_receiver: Receiver<Command>,
    generator: Option<(Box<dyn Generator>, usize)>,
    sample_sender: Sender<Vec<f32>>,
    send_counter: u32,
}

pub fn new_tracker(
    sample_frequency: i32,
    beats_per_minute: i32,
    command_receiver: Receiver<Command>,
    sample_sender: Sender<Vec<f32>>,
) -> Tracker {
    return Tracker {
        sample_frequency: sample_frequency,
        _beats_per_minute: beats_per_minute,
        command_receiver: command_receiver,
        generator: None,
        sample_sender: sample_sender,
        send_counter: 0,
    };
}

// TODO add metrics

impl<'a> AudioCallback for Tracker {
    type Channel = f32;

    fn callback(&mut self, out: &mut [f32]) {
        for x in out.iter_mut() {
            *x = 0.0;
        }
        let mut generated = 0;
        let mut recv_allowed = true;
        while generated < out.len() && recv_allowed {
            if self.generator.is_none() && recv_allowed {
                recv_allowed = false;
                match self.command_receiver.try_recv() {
                    Ok(Command::PlayOnce { expr, beat }) => {
                        println!(
                            "Received command to play once at {} with expression {:?}",
                            beat, expr
                        );
                        let generator = from_expr(self.sample_frequency, expr);
                        match generator {
                            None => println!("Failed to convert expression to generator"),
                            Some(generator) => {
                                self.generator = Some((generator, 0));
                            }
                        }
                        recv_allowed = true;
                    }
                    Err(_) => {}
                }
            }
            match &self.generator {
                Some((generator, offset)) => {
                    match generator.samples() {
                        Some(samples) if samples - offset < out.len() => {
                            let tmp = &mut out[generated..samples - offset];
                            generator.generate(*offset, tmp);
                            generated += tmp.len();
                            self.generator = None;
                            recv_allowed = true;
                        }
                        _ => {
                            // Unbounded generator or enough samples to fill the buffer
                            let tmp = &mut out[generated..];
                            generator.generate(*offset, tmp);
                            generated += tmp.len();
                            self.generator.as_mut().unwrap().1 = offset + tmp.len();
                        }
                    }
                }
                None => (),
            }
        }

        if self.send_counter == 0 {
            let mut copy: Vec<f32> = Vec::with_capacity(out.len());
            out.clone_into(&mut copy);
            self.sample_sender.send(copy).unwrap();
            self.send_counter = 5;
        } else {
            self.send_counter -= 1;
        }
    }

    // TODO need some sort of way to signal done
}

#[cfg(test)]
mod tests {
    use super::*;

    struct ConstGenerator {
        value: f32,
    }

    impl Generator for ConstGenerator {
        fn generate(&self, _offset: usize, out: &mut [f32]) {
            for x in out.iter_mut() {
                *x = self.value;
            }
        }
    }
    fn const_gen(value: f32) -> Box<dyn Generator> {
        return Box::new(ConstGenerator { value });
    }
    fn finite_const_gen(value: f32, samples: usize, next_offset: usize) -> Box<dyn Generator> {
        return Box::new(Seq {
            generator: Box::new(Fin {
                samples,
                generator: const_gen(value),
            }),
            next_offset,
        });
    }

    fn generate_samples(generator: &dyn Generator, size: usize, out: &mut [f32]) {
        for n in 0..out.len() / size {
            generator.generate(n * size, &mut out[n * size..(n + 1) * size]);
        }
    }

    #[test]
    fn test_chord() {
        let chord_gen = Chord {
            generators: vec![
                finite_const_gen(1.0, 5, 5),
                finite_const_gen(0.5, 10, 10),
                finite_const_gen(0.25, 15, 15),
                finite_const_gen(0.125, 8, 8),
            ],
            samples: Some(16),
            next_offset: Some(16),
        };
        let desired = vec![
            1.875, 1.875, 1.875, 1.875, 1.875, 0.875, 0.875, 0.875, 0.75, 0.75, 0.25, 0.25, 0.25,
            0.25, 0.25, 0.0,
        ];
        for size in [1, 2, 4, 8, 16] {
            let mut out = vec![0.0; 16];
            println!("Generating chord with buffer size {}", size);
            generate_samples(&chord_gen, size, &mut out);
            assert_eq!(out, desired);
        }
    }

    #[test]
    fn test_sequence() {
        let seq_gen = Sequence {
            generators: vec![
                finite_const_gen(1.0, 5, 1),
                finite_const_gen(0.5, 10, 8),
                finite_const_gen(0.25, 5, 5),
            ],
            samples: Some(16),
            next_offset: Some(16),
        };
        let desired = vec![
            1.0, 1.5, 1.5, 1.5, 1.5, 0.5, 0.5, 0.5, 0.5, 0.75, 0.75, 0.25, 0.25, 0.25, 0.0, 0.0,
        ];
        for size in [1, 2, 4, 8, 16] {
            let mut out = vec![0.0; 16];
            println!("Generating sequence with buffer size {}", size);
            generate_samples(&seq_gen, size, &mut out);
            assert_eq!(out, desired);
        }
    }
}
