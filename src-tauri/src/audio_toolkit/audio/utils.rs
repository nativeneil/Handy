use anyhow::Result;
use hound::{WavReader, WavSpec, WavWriter};
use log::debug;
use std::path::Path;

/// Read a WAV file and return normalised f32 samples.
pub fn read_wav_samples<P: AsRef<Path>>(file_path: P) -> Result<Vec<f32>> {
    let reader = WavReader::open(file_path.as_ref())?;
    let samples = reader
        .into_samples::<i16>()
        .map(|s| s.map(|v| v as f32 / i16::MAX as f32))
        .collect::<Result<Vec<f32>, _>>()?;
    Ok(samples)
}

/// Verify a WAV file by reading it back and checking the sample count.
pub fn verify_wav_file<P: AsRef<Path>>(file_path: P, expected_samples: usize) -> Result<()> {
    let reader = WavReader::open(file_path.as_ref())?;
    let actual_samples = reader.len() as usize;
    if actual_samples != expected_samples {
        anyhow::bail!(
            "WAV sample count mismatch: expected {}, got {}",
            expected_samples,
            actual_samples
        );
    }
    Ok(())
}

/// Handy's recording format: 16 kHz mono 16-bit PCM.
const WAV_SPEC: WavSpec = WavSpec {
    channels: 1,
    sample_rate: 16000,
    bits_per_sample: 16,
    sample_format: hound::SampleFormat::Int,
};

fn write_wav_samples<W: std::io::Write + std::io::Seek>(
    mut writer: WavWriter<W>,
    samples: &[f32],
) -> Result<()> {
    for sample in samples {
        // `as` saturates, so out-of-range samples clip instead of wrapping.
        writer.write_sample((sample * i16::MAX as f32) as i16)?;
    }
    writer.finalize()?;
    Ok(())
}

/// Encode audio samples as an in-memory WAV file (e.g. for upload).
pub fn encode_wav_bytes(samples: &[f32]) -> Result<Vec<u8>> {
    let mut cursor = std::io::Cursor::new(Vec::with_capacity(samples.len() * 2 + 44));
    write_wav_samples(WavWriter::new(&mut cursor, WAV_SPEC)?, samples)?;
    Ok(cursor.into_inner())
}

/// Save audio samples as a WAV file
pub fn save_wav_file<P: AsRef<Path>>(file_path: P, samples: &[f32]) -> Result<()> {
    write_wav_samples(WavWriter::create(file_path.as_ref(), WAV_SPEC)?, samples)?;
    debug!("Saved WAV file: {:?}", file_path.as_ref());
    Ok(())
}
