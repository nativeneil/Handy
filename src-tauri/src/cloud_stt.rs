//! Cloud speech-to-text: send a finished recording to a hosted transcription
//! API instead of the local model. Selected by `transcription_provider_id`;
//! "local" (the default) keeps the on-device pipeline.

use crate::settings::AppSettings;
use anyhow::{anyhow, Context, Result};
use base64::Engine as _;
use log::{debug, info};
use reqwest::blocking::{multipart, Client};
use serde::Serialize;
use serde_json::{json, Value};
use specta::Type;
use std::io::Cursor;
use std::time::{Duration, Instant};

pub const LOCAL_PROVIDER_ID: &str = "local";

const REQUEST_TIMEOUT: Duration = Duration::from_secs(120);

/// Wire protocol a cloud provider speaks.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Type)]
#[serde(rename_all = "snake_case")]
pub enum CloudSttApi {
    /// `POST {base}/audio/transcriptions`, multipart (OpenAI, Groq, Mistral,
    /// and any compatible server).
    OpenAi,
    Deepgram,
    ElevenLabs,
    Gemini,
}

#[derive(Clone, Debug, Serialize, Type)]
pub struct CloudSttProvider {
    pub id: &'static str,
    pub label: &'static str,
    pub api: CloudSttApi,
    pub base_url: &'static str,
    pub allow_base_url_edit: bool,
    pub default_model: &'static str,
    /// Suggested models for the picker; any model id can be typed.
    pub models: Vec<&'static str>,
}

pub fn providers() -> Vec<CloudSttProvider> {
    vec![
        CloudSttProvider {
            id: "openai",
            label: "OpenAI",
            api: CloudSttApi::OpenAi,
            base_url: "https://api.openai.com/v1",
            allow_base_url_edit: false,
            default_model: "gpt-4o-transcribe",
            models: vec!["gpt-4o-transcribe", "gpt-4o-mini-transcribe", "whisper-1"],
        },
        CloudSttProvider {
            id: "groq",
            label: "Groq",
            api: CloudSttApi::OpenAi,
            base_url: "https://api.groq.com/openai/v1",
            allow_base_url_edit: false,
            default_model: "whisper-large-v3-turbo",
            models: vec!["whisper-large-v3-turbo", "whisper-large-v3"],
        },
        CloudSttProvider {
            id: "deepgram",
            label: "Deepgram",
            api: CloudSttApi::Deepgram,
            base_url: "https://api.deepgram.com/v1",
            allow_base_url_edit: false,
            default_model: "nova-3",
            models: vec!["nova-3", "nova-2"],
        },
        CloudSttProvider {
            id: "elevenlabs",
            label: "ElevenLabs",
            api: CloudSttApi::ElevenLabs,
            base_url: "https://api.elevenlabs.io/v1",
            allow_base_url_edit: false,
            default_model: "scribe_v1",
            models: vec!["scribe_v1"],
        },
        CloudSttProvider {
            id: "gemini",
            label: "Google Gemini",
            api: CloudSttApi::Gemini,
            base_url: "https://generativelanguage.googleapis.com/v1beta",
            allow_base_url_edit: false,
            default_model: "gemini-2.5-flash",
            models: vec![
                "gemini-2.5-flash",
                "gemini-2.5-flash-lite",
                "gemini-2.5-pro",
            ],
        },
        CloudSttProvider {
            id: "mistral",
            label: "Mistral (Voxtral)",
            api: CloudSttApi::OpenAi,
            base_url: "https://api.mistral.ai/v1",
            allow_base_url_edit: false,
            default_model: "voxtral-mini-latest",
            models: vec!["voxtral-mini-latest"],
        },
        CloudSttProvider {
            id: "custom",
            label: "Custom (OpenAI-compatible)",
            api: CloudSttApi::OpenAi,
            base_url: "http://localhost:8000/v1",
            allow_base_url_edit: true,
            default_model: "whisper-1",
            models: vec![],
        },
    ]
}

pub fn provider(id: &str) -> Option<CloudSttProvider> {
    providers().into_iter().find(|p| p.id == id)
}

/// The cloud provider the settings select, or None for local transcription.
pub fn active_provider(settings: &AppSettings) -> Option<CloudSttProvider> {
    if settings.transcription_provider_id == LOCAL_PROVIDER_ID {
        return None;
    }
    provider(&settings.transcription_provider_id)
}

pub fn uses_cloud(settings: &AppSettings) -> bool {
    active_provider(settings).is_some()
}

/// Everything one request needs, resolved from settings.
struct Request {
    provider: CloudSttProvider,
    base_url: String,
    api_key: String,
    model: String,
    language: Option<String>,
    vocabulary: Vec<String>,
}

impl Request {
    fn from_settings(provider: CloudSttProvider, settings: &AppSettings) -> Result<Self> {
        let api_key = settings
            .transcription_api_keys
            .get(provider.id)
            .cloned()
            .unwrap_or_default();
        if api_key.trim().is_empty() && provider.id != "custom" {
            return Err(anyhow!(
                "No API key set for {} transcription. Add one in Settings.",
                provider.label
            ));
        }
        let model = settings
            .transcription_models
            .get(provider.id)
            .map(|m| m.trim().to_string())
            .filter(|m| !m.is_empty())
            .unwrap_or_else(|| provider.default_model.to_string());
        let base_url = settings
            .transcription_base_urls
            .get(provider.id)
            .filter(|_| provider.allow_base_url_edit)
            .map(|u| u.trim().to_string())
            .filter(|u| !u.is_empty())
            .unwrap_or_else(|| provider.base_url.to_string());
        let language = match settings.selected_language.as_str() {
            "" | "auto" => None,
            // Handy uses zh-Hans/zh-Hant; cloud APIs want ISO-639-1.
            lang => Some(lang.split('-').next().unwrap_or(lang).to_string()),
        };
        Ok(Self {
            base_url: base_url.trim_end_matches('/').to_string(),
            api_key: api_key.trim().to_string(),
            model,
            language,
            vocabulary: settings.custom_words.clone(),
            provider,
        })
    }
}

/// Transcribe 16 kHz mono f32 PCM with the selected cloud provider.
///
/// Runs on its own thread: callers may be on an async runtime worker, where a
/// blocking HTTP client must not be created or dropped.
pub fn transcribe(settings: &AppSettings, samples: &[f32]) -> Result<String> {
    let provider =
        active_provider(settings).ok_or_else(|| anyhow!("No cloud provider selected"))?;
    let request = Request::from_settings(provider, settings)?;
    let wav = encode_wav(samples)?;
    let started = Instant::now();
    let label = request.provider.label;
    let model = request.model.clone();

    let text = std::thread::spawn(move || send(&request, wav))
        .join()
        .map_err(|_| anyhow!("Cloud transcription thread panicked"))??;

    info!(
        "Cloud transcription via {} ({}) took {:.2}s",
        label,
        model,
        started.elapsed().as_secs_f64()
    );
    Ok(text.trim().to_string())
}

fn send(req: &Request, wav: Vec<u8>) -> Result<String> {
    let client = Client::builder()
        .timeout(REQUEST_TIMEOUT)
        .build()
        .context("Failed to build HTTP client")?;
    debug!(
        "Cloud STT request: provider={} model={} bytes={}",
        req.provider.id,
        req.model,
        wav.len()
    );
    match req.provider.api {
        CloudSttApi::OpenAi => send_openai(&client, req, wav),
        CloudSttApi::Deepgram => send_deepgram(&client, req, wav),
        CloudSttApi::ElevenLabs => send_elevenlabs(&client, req, wav),
        CloudSttApi::Gemini => send_gemini(&client, req, wav),
    }
}

fn wav_part(wav: Vec<u8>) -> Result<multipart::Part> {
    multipart::Part::bytes(wav)
        .file_name("audio.wav")
        .mime_str("audio/wav")
        .context("Invalid audio MIME type")
}

fn send_openai(client: &Client, req: &Request, wav: Vec<u8>) -> Result<String> {
    let mut form = multipart::Form::new()
        .part("file", wav_part(wav)?)
        .text("model", req.model.clone())
        .text("response_format", "json");
    if let Some(lang) = &req.language {
        form = form.text("language", lang.clone());
    }
    if !req.vocabulary.is_empty() {
        form = form.text(
            "prompt",
            format!("Vocabulary: {}.", req.vocabulary.join(", ")),
        );
    }
    let mut builder = client
        .post(format!("{}/audio/transcriptions", req.base_url))
        .multipart(form);
    if !req.api_key.is_empty() {
        builder = builder.bearer_auth(&req.api_key);
    }
    let body = read_json(builder.send())?;
    string_at(&body, &["text"])
}

fn send_deepgram(client: &Client, req: &Request, wav: Vec<u8>) -> Result<String> {
    let mut query: Vec<(&str, String)> = vec![
        ("model", req.model.clone()),
        ("smart_format", "true".into()),
        ("punctuate", "true".into()),
    ];
    match &req.language {
        Some(lang) => query.push(("language", lang.clone())),
        None => query.push(("detect_language", "true".into())),
    }
    for word in &req.vocabulary {
        query.push(("keyterm", word.clone()));
    }
    let builder = client
        .post(format!("{}/listen", req.base_url))
        .query(&query)
        .header("Authorization", format!("Token {}", req.api_key))
        .header("Content-Type", "audio/wav")
        .body(wav);
    let body = read_json(builder.send())?;
    string_at(
        &body,
        &[
            "results",
            "channels",
            "0",
            "alternatives",
            "0",
            "transcript",
        ],
    )
}

fn send_elevenlabs(client: &Client, req: &Request, wav: Vec<u8>) -> Result<String> {
    let mut form = multipart::Form::new()
        .part("file", wav_part(wav)?)
        .text("model_id", req.model.clone())
        .text("tag_audio_events", "false");
    if let Some(lang) = &req.language {
        form = form.text("language_code", lang.clone());
    }
    let builder = client
        .post(format!("{}/speech-to-text", req.base_url))
        .header("xi-api-key", &req.api_key)
        .multipart(form);
    let body = read_json(builder.send())?;
    string_at(&body, &["text"])
}

fn send_gemini(client: &Client, req: &Request, wav: Vec<u8>) -> Result<String> {
    let mut prompt = String::from(
        "Transcribe this audio verbatim. Output only the transcript text, with \
         natural punctuation and capitalization. No commentary, labels, or timestamps. \
         If there is no speech, output nothing.",
    );
    if let Some(lang) = &req.language {
        prompt.push_str(&format!(" The speech is in language code '{}'.", lang));
    }
    if !req.vocabulary.is_empty() {
        prompt.push_str(&format!(
            " Spell these terms exactly when they occur: {}.",
            req.vocabulary.join(", ")
        ));
    }
    let payload = json!({
        "contents": [{
            "parts": [
                { "text": prompt },
                { "inline_data": {
                    "mime_type": "audio/wav",
                    "data": base64::engine::general_purpose::STANDARD.encode(wav),
                }}
            ]
        }],
        "generationConfig": { "temperature": 0 }
    });
    let builder = client
        .post(format!(
            "{}/models/{}:generateContent",
            req.base_url, req.model
        ))
        .header("x-goog-api-key", &req.api_key)
        .json(&payload);
    let body = read_json(builder.send())?;
    // A response may split text across parts.
    let parts = body
        .pointer("/candidates/0/content/parts")
        .and_then(Value::as_array)
        .ok_or_else(|| {
            anyhow!(
                "Unexpected Gemini response: {}",
                truncate(&body.to_string())
            )
        })?;
    Ok(parts
        .iter()
        .filter_map(|p| p.get("text").and_then(Value::as_str))
        .collect::<String>())
}

fn read_json(result: reqwest::Result<reqwest::blocking::Response>) -> Result<Value> {
    let response =
        result.map_err(|e| anyhow!("Cloud transcription request failed: {}", e.without_url()))?;
    let status = response.status();
    let text = response
        .text()
        .context("Failed to read cloud transcription response")?;
    if !status.is_success() {
        return Err(anyhow!(
            "Cloud transcription failed ({}): {}",
            status,
            truncate(&text)
        ));
    }
    serde_json::from_str(&text)
        .with_context(|| format!("Cloud transcription returned non-JSON: {}", truncate(&text)))
}

/// Follow `path` (object keys or array indices) to a string.
fn string_at(body: &Value, path: &[&str]) -> Result<String> {
    let pointer = format!("/{}", path.join("/"));
    body.pointer(&pointer)
        .and_then(Value::as_str)
        .map(str::to_string)
        .ok_or_else(|| anyhow!("Unexpected response shape: {}", truncate(&body.to_string())))
}

fn truncate(s: &str) -> String {
    const MAX: usize = 500;
    if s.len() <= MAX {
        s.to_string()
    } else {
        let mut end = MAX;
        while !s.is_char_boundary(end) {
            end -= 1;
        }
        format!("{}…", &s[..end])
    }
}

/// 16-bit PCM WAV, matching what Handy saves to history.
fn encode_wav(samples: &[f32]) -> Result<Vec<u8>> {
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate: 16000,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let mut cursor = Cursor::new(Vec::with_capacity(samples.len() * 2 + 44));
    {
        let mut writer = hound::WavWriter::new(&mut cursor, spec)?;
        for sample in samples {
            writer.write_sample((sample.clamp(-1.0, 1.0) * i16::MAX as f32) as i16)?;
        }
        writer.finalize()?;
    }
    Ok(cursor.into_inner())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wav_has_header_and_samples() {
        let wav = encode_wav(&[0.0; 1600]).unwrap();
        assert_eq!(&wav[0..4], b"RIFF");
        assert_eq!(wav.len(), 44 + 3200);
    }

    #[test]
    fn provider_ids_are_unique_and_not_local() {
        let ids: Vec<_> = providers().iter().map(|p| p.id).collect();
        let mut sorted = ids.clone();
        sorted.sort();
        sorted.dedup();
        assert_eq!(ids.len(), sorted.len());
        assert!(!ids.contains(&LOCAL_PROVIDER_ID));
    }

    #[test]
    fn string_at_walks_arrays() {
        let body = json!({"results":{"channels":[{"alternatives":[{"transcript":"hi"}]}]}});
        assert_eq!(
            string_at(
                &body,
                &[
                    "results",
                    "channels",
                    "0",
                    "alternatives",
                    "0",
                    "transcript"
                ]
            )
            .unwrap(),
            "hi"
        );
    }
}
