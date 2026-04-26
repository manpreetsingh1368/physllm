//! audio.rs — Cross-platform audio capture and playback using cpal.
//!
//! Supports:
//!   - Linux:   ALSA / PipeWire / PulseAudio
//!   - Windows: WASAPI
//!   - macOS:   CoreAudio
//!
//! Audio flow:
//!   Device → stream → ring buffer → resampler → 16kHz mono f32 → VAD/Whisper

use crate::{Result, VoiceError};
use cpal::{
    traits::{DeviceTrait, HostTrait, StreamTrait},
    Device, Host, SampleFormat, SampleRate, StreamConfig,
};
use rubato::{Resampler, SincFixedIn, SincInterpolationType, SincInterpolationParameters, WindowFunction};
use std::sync::{Arc, Mutex};
use tokio::sync::mpsc;
use tracing::{info, debug, warn};

pub const TARGET_SAMPLE_RATE: u32 = 16_000;   // Whisper expects 16kHz
pub const TARGET_CHANNELS:    u16 = 1;         // mono
pub const CHUNK_DURATION_MS:  u32 = 30;        // 30ms chunks (good for Opus + VAD)

/// A chunk of raw audio (f32 mono, 16kHz).
#[derive(Debug, Clone)]
pub struct AudioChunk {
    pub samples:     Vec<f32>,
    pub sample_rate: u32,
    pub channels:    u16,
    pub timestamp_ms: u64,
}

impl AudioChunk {
    pub fn duration_ms(&self) -> u64 {
        (self.samples.len() as u64 * 1000) / self.sample_rate as u64
    }

    pub fn is_silent(&self, threshold: f32) -> bool {
        let rms = (self.samples.iter().map(|&s| s * s).sum::<f32>()
            / self.samples.len() as f32).sqrt();
        rms < threshold
    }
}

/// Audio device manager — wraps cpal host/device selection.
pub struct AudioDevices {
    host:       Host,
    pub input:  Option<Device>,
    pub output: Option<Device>,
}

impl AudioDevices {
    /// Open the default audio devices.
    pub fn default_devices() -> Result<Self> {
        let host = cpal::default_host();
        info!("Audio host: {}", host.id().name());

        let input = host.default_input_device();
        let output = host.default_output_device();

        if let Some(ref d) = input  { info!("Input device:  {}", d.name().unwrap_or_default()); }
        if let Some(ref d) = output { info!("Output device: {}", d.name().unwrap_or_default()); }

        Ok(Self { host, input, output })
    }

    /// List all available input devices.
    pub fn list_inputs(&self) -> Vec<String> {
        self.host.input_devices()
            .map(|devs| devs.filter_map(|d| d.name().ok()).collect())
            .unwrap_or_default()
    }

    /// Get the supported input config closest to 16kHz mono.
    pub fn best_input_config(&self) -> Result<(StreamConfig, SampleFormat)> {
        let device = self.input.as_ref()
            .ok_or_else(|| VoiceError::AudioDevice("No input device found".into()))?;

        let configs = device.supported_input_configs()
            .map_err(|e| VoiceError::AudioDevice(e.to_string()))?;

        // Prefer 16kHz mono f32, fall back to anything supported
        let mut best: Option<cpal::SupportedStreamConfigRange> = None;
        for cfg in configs {
            if cfg.channels() == 1 && cfg.sample_format() == SampleFormat::F32 {
                best = Some(cfg);
                break;
            }
            if best.is_none() { best = Some(cfg); }
        }

        let cfg = best.ok_or_else(|| VoiceError::AudioDevice("No supported input config".into()))?;
        let rate = if cfg.min_sample_rate().0 <= 16_000 && cfg.max_sample_rate().0 >= 16_000 {
            SampleRate(16_000)
        } else {
            cfg.max_sample_rate()
        };
        let stream_cfg = cfg.with_sample_rate(rate).config();
        let fmt = cpal::SampleFormat::F32; // we convert everything to f32
        Ok((stream_cfg, fmt))
    }
}

/// Microphone capture — streams audio chunks to a channel.
pub struct MicCapture {
    stream:      cpal::Stream,
    pub rx:      mpsc::Receiver<AudioChunk>,
    device_rate: u32,
    device_ch:   u16,
    resampler:   Option<Arc<Mutex<SincFixedIn<f32>>>>,
}

impl MicCapture {
    /// Start capturing from the default microphone.
    /// Returns a `MicCapture` whose `rx` channel yields 16kHz mono chunks.
    pub fn start(devices: &AudioDevices) -> Result<Self> {
        let device = devices.input.as_ref()
            .ok_or_else(|| VoiceError::AudioDevice("No microphone".into()))?;
        let (cfg, _fmt) = devices.best_input_config()?;

        let device_rate = cfg.sample_rate.0;
        let device_ch   = cfg.channels;
        info!("Mic: {} Hz, {} ch", device_rate, device_ch);

        let (tx, rx) = mpsc::channel::<AudioChunk>(64);
        let chunk_size = (device_rate * CHUNK_DURATION_MS / 1000) as usize * device_ch as usize;
        let buffer: Arc<Mutex<Vec<f32>>> = Arc::new(Mutex::new(Vec::with_capacity(chunk_size * 2)));
        let tx_clone = tx.clone();
        let buf_clone = buffer.clone();

        let ts = Arc::new(Mutex::new(0u64));
        let ts_clone = ts.clone();
        let rate = device_rate;
        let ch   = device_ch;

        let stream = device.build_input_stream(
            &cfg,
            move |data: &[f32], _| {
                let mut buf = buf_clone.lock().unwrap();
                buf.extend_from_slice(data);

                while buf.len() >= chunk_size {
                    let raw: Vec<f32> = buf.drain(..chunk_size).collect();
                    // Downmix to mono if stereo
                    let mono: Vec<f32> = if ch > 1 {
                        raw.chunks(ch as usize)
                            .map(|frame| frame.iter().sum::<f32>() / ch as f32)
                            .collect()
                    } else { raw };

                    let mut ts_guard = ts_clone.lock().unwrap();
                    let chunk = AudioChunk {
                        samples:     mono,
                        sample_rate: rate,
                        channels:    1,
                        timestamp_ms: *ts_guard,
                    };
                    *ts_guard += CHUNK_DURATION_MS as u64;
                    let _ = tx_clone.try_send(chunk);
                }
            },
            |err| warn!("Mic error: {err}"),
            None,
        ).map_err(|e| VoiceError::AudioDevice(e.to_string()))?;

        stream.play().map_err(|e| VoiceError::AudioDevice(e.to_string()))?;

        // Build resampler if device rate != 16kHz
        let resampler = if device_rate != TARGET_SAMPLE_RATE {
            let params = SincInterpolationParameters {
                sinc_len: 256,
                f_cutoff: 0.95,
                interpolation: SincInterpolationType::Linear,
                oversampling_factor: 256,
                window: WindowFunction::BlackmanHarris2,
            };
            let r = SincFixedIn::<f32>::new(
                TARGET_SAMPLE_RATE as f64 / device_rate as f64,
                2.0,
                params,
                chunk_size / device_ch as usize,
                1, // mono
            ).map_err(|e| VoiceError::AudioDevice(e.to_string()))?;
            Some(Arc::new(Mutex::new(r)))
        } else { None };

        Ok(Self { stream, rx, device_rate, device_ch, resampler })
    }

    /// Resample a chunk to 16kHz if needed.
    pub fn resample(&self, chunk: &AudioChunk) -> Result<AudioChunk> {
        if self.resampler.is_none() || chunk.sample_rate == TARGET_SAMPLE_RATE {
            return Ok(chunk.clone());
        }
        let mut rs = self.resampler.as_ref().unwrap().lock().unwrap();
        let waves = vec![chunk.samples.clone()];
        let out = rs.process(&waves, None)
            .map_err(|e| VoiceError::AudioDevice(format!("Resample: {e}")))?;
        Ok(AudioChunk {
            samples:     out.into_iter().next().unwrap_or_default(),
            sample_rate: TARGET_SAMPLE_RATE,
            channels:    1,
            timestamp_ms: chunk.timestamp_ms,
        })
    }
}

/// Speaker playback — plays audio chunks via the output device.
pub struct Speaker {
    tx:     mpsc::Sender<Vec<f32>>,
    stream: cpal::Stream,
}

impl Speaker {
    pub fn start(devices: &AudioDevices) -> Result<Self> {
        let device = devices.output.as_ref()
            .ok_or_else(|| VoiceError::AudioDevice("No speaker".into()))?;
        let cfg = device.default_output_config()
            .map_err(|e| VoiceError::AudioDevice(e.to_string()))?;
        let stream_cfg = cfg.config();

        let (tx, mut rx) = mpsc::channel::<Vec<f32>>(128);
        let playback_buf: Arc<Mutex<Vec<f32>>> = Arc::new(Mutex::new(Vec::new()));
        let pb_clone = playback_buf.clone();

        // Feed received audio into playback buffer
        tokio::spawn(async move {
            while let Some(chunk) = rx.recv().await {
                let mut buf = pb_clone.lock().unwrap();
                buf.extend(chunk);
            }
        });

        let pb = playback_buf.clone();
        let stream = device.build_output_stream(
            &stream_cfg,
            move |out: &mut [f32], _| {
                let mut buf = pb.lock().unwrap();
                let n = out.len().min(buf.len());
                out[..n].copy_from_slice(&buf[..n]);
                buf.drain(..n);
                if n < out.len() {
                    out[n..].fill(0.0); // silence for unfilled frames
                }
            },
            |err| warn!("Speaker error: {err}"),
            None,
        ).map_err(|e| VoiceError::AudioDevice(e.to_string()))?;

        stream.play().map_err(|e| VoiceError::AudioDevice(e.to_string()))?;
        Ok(Self { tx, stream })
    }

    /// Send audio samples to the speaker.
    pub async fn play(&self, samples: Vec<f32>) {
        let _ = self.tx.send(samples).await;
    }

    /// Play a WAV file.
    pub async fn play_wav(&self, path: &str) -> Result<()> {
        let mut reader = hound::WavReader::open(path)
            .map_err(|e| VoiceError::AudioDevice(e.to_string()))?;
        let samples: Vec<f32> = reader.samples::<i16>()
            .filter_map(|s| s.ok())
            .map(|s| s as f32 / i16::MAX as f32)
            .collect();
        self.play(samples).await;
        Ok(())
    }
}

/// Save audio samples to a WAV file.
pub fn save_wav(samples: &[f32], sample_rate: u32, path: &str) -> Result<()> {
    let spec = hound::WavSpec {
        channels:        1,
        sample_rate,
        bits_per_sample: 16,
        sample_format:   hound::SampleFormat::Int,
    };
    let mut writer = hound::WavWriter::create(path, spec)
        .map_err(|e| VoiceError::Io(std::io::Error::new(std::io::ErrorKind::Other, e.to_string())))?;
    for &s in samples {
        let s_i16 = (s * i16::MAX as f32).clamp(i16::MIN as f32, i16::MAX as f32) as i16;
        writer.write_sample(s_i16)
            .map_err(|e| VoiceError::Io(std::io::Error::new(std::io::ErrorKind::Other, e.to_string())))?;
    }
    writer.finalize()
        .map_err(|e| VoiceError::Io(std::io::Error::new(std::io::ErrorKind::Other, e.to_string())))?;
    Ok(())
}
