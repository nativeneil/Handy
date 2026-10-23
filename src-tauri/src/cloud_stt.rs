//! Cloud speech-to-text: send a finished recording to a hosted transcription
//! API instead of the local model. Selected by `transcription_provider_id`;
//! "local" (the default) keeps the on-device pipeline.
//!
//! Adding a provider: append an entry to [`PROVIDERS`]. If it speaks an
//! existing [`CloudSttApi`] (most are OpenAI-compatible) that is all; set
//! `vocab` to say how custom words are sent. Only a new wire protocol needs a
//! new `CloudSttApi` variant and `send_*` function.

use crate::managers::model::base_language;
use crate::settings::AppSettings;
use anyhow::{anyhow, Context, Result};
use base64::Engine as _;
use log::{debug, error, info};
use reqwest::blocking::{multipart, Client, RequestBuilder};
use serde::Serialize;
use serde_json::{json, Value};
use specta::Type;
use std::sync::LazyLock;
use std::time::{Duration, Instant};

pub const LOCAL_PROVIDER_ID: &str = "local";

const REQUEST_TIMEOUT: Duration = Duration::from_secs(120);
const PREWARM_TIMEOUT: Duration = Duration::from_secs(5);
/// Upper bound on waiting for an async (submit-then-poll) job.
const POLL_TIMEOUT: Duration = Duration::from_secs(120);
const POLL_INTERVAL: Duration = Duration::from_millis(150);

/// One client for the app's lifetime so dictations reuse pooled connections
/// (no DNS/TCP/TLS setup after key release). First touched off the async
/// runtime, and never dropped, so the blocking client is safe here.
static CLIENT: LazyLock<Client> = LazyLock::new(|| {
    Client::builder()
        .timeout(REQUEST_TIMEOUT)
        .build()
        .expect("Failed to build HTTP client")
});

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
    /// xAI Grok Voice Transcribe: `POST {base}/stt`, multipart.
    Xai,
    /// AssemblyAI: upload, create transcript, poll.
    AssemblyAi,
    /// Azure Speech fast transcription (MAI-Transcribe) on a user's resource.
    AzureSpeech,
    /// OpenRouter's unified `audio/transcriptions`, JSON with base64 audio.
    OpenRouter,
}

/// How custom words reach a multipart provider.
#[derive(Clone, Copy, Debug)]
enum Vocab {
    /// Folded into a free-text `prompt` field.
    Prompt,
    /// One form field per word, under this name.
    Repeated(&'static str),
    /// Handled by the API's own adapter (JSON body, query string), or unsupported.
    Native,
}

#[derive(Clone, Debug, Serialize, Type)]
pub struct CloudSttProvider {
    pub id: &'static str,
    pub label: &'static str,
    pub api: CloudSttApi,
    /// Empty when the user must supply it (`allow_base_url_edit`).
    pub base_url: &'static str,
    pub allow_base_url_edit: bool,
    /// Shown in the empty Base URL field.
    pub base_url_placeholder: &'static str,
    pub requires_api_key: bool,
    /// Empty when the API has no model choice.
    pub default_model: &'static str,
    /// Suggested models for the picker; any model id can be typed.
    pub models: &'static [&'static str],
    #[serde(skip)]
    #[specta(skip)]
    vocab: Vocab,
}

const fn hosted(
    id: &'static str,
    label: &'static str,
    api: CloudSttApi,
    base_url: &'static str,
    default_model: &'static str,
    models: &'static [&'static str],
    vocab: Vocab,
) -> CloudSttProvider {
    CloudSttProvider {
        id,
        label,
        api,
        base_url,
        allow_base_url_edit: false,
        base_url_placeholder: "",
        requires_api_key: true,
        default_model,
        models,
        vocab,
    }
}

/// Ordered roughly best-first for dictation.
pub static PROVIDERS: &[CloudSttProvider] = &[
    hosted(
        "elevenlabs",
        "ElevenLabs Scribe",
        CloudSttApi::ElevenLabs,
        "https://api.elevenlabs.io/v1",
        "scribe_v2",
        &["scribe_v2", "scribe_v1"],
        Vocab::Repeated("keyterms"),
    ),
    hosted(
        "mistral",
        "Mistral Voxtral",
        CloudSttApi::OpenAi,
        "https://api.mistral.ai/v1",
        "voxtral-mini-latest",
        &["voxtral-mini-latest", "voxtral-mini-transcribe-3-0"],
        Vocab::Repeated("context_bias"),
    ),
    hosted(
        "xai",
        "xAI Grok Voice Transcribe",
        CloudSttApi::Xai,
        "https://api.x.ai/v1",
        "",
        &[],
        Vocab::Repeated("keyterm"),
    ),
    CloudSttProvider {
        base_url: "",
        allow_base_url_edit: true,
        base_url_placeholder: "https://NAME.cognitiveservices.azure.com",
        ..hosted(
            "azure_mai",
            "Microsoft MAI-Transcribe (Azure Speech)",
            CloudSttApi::AzureSpeech,
            "",
            "MAI-Transcribe-2",
            &["MAI-Transcribe-2"],
            Vocab::Native,
        )
    },
    hosted(
        "gemini",
        "Google Gemini",
        CloudSttApi::Gemini,
        "https://generativelanguage.googleapis.com/v1beta",
        "gemini-3.5-transcribe",
        &[
            "gemini-3.5-transcribe",
            "gemini-2.5-flash",
            "gemini-2.5-flash-lite",
        ],
        Vocab::Native,
    ),
    hosted(
        "assemblyai",
        "AssemblyAI",
        CloudSttApi::AssemblyAi,
        "https://api.assemblyai.com/v2",
        "universal-3-5-pro",
        &["universal-3-5-pro", "universal-3-pro", "universal-2"],
        Vocab::Native,
    ),
    hosted(
        "openrouter",
        "OpenRouter (incl. Meta Muse Voice)",
        CloudSttApi::OpenRouter,
        "https://openrouter.ai/api/v1",
        "meta/muse-voice-transcribe-1.0",
        &[
            "meta/muse-voice-transcribe-1.0",
            "mistralai/voxtral-mini-transcribe-3-0",
            "google/gemini-3.5-transcribe",
            "openai/gpt-transcribe",
        ],
        Vocab::Native,
    ),
    hosted(
        "openai",
        "OpenAI",
        CloudSttApi::OpenAi,
        "https://api.openai.com/v1",
        "gpt-transcribe",
        &[
            "gpt-transcribe",
            "gpt-4o-transcribe",
            "gpt-4o-mini-transcribe",
            "whisper-1",
        ],
        Vocab::Prompt,
    ),
    hosted(
        "deepgram",
        "Deepgram",
        CloudSttApi::Deepgram,
        "https://api.deepgram.com/v1",
        "nova-3",
        &["nova-3", "nova-2"],
        Vocab::Native,
    ),
    hosted(
        "groq",
        "Groq",
        CloudSttApi::OpenAi,
        "https://api.groq.com/openai/v1",
        "whisper-large-v3-turbo",
        &["whisper-large-v3-turbo", "whisper-large-v3"],
        Vocab::Prompt,
    ),
    CloudSttProvider {
        allow_base_url_edit: true,
        base_url_placeholder: "http://localhost:8000/v1",
        requires_api_key: false,
        ..hosted(
            "custom",
            "Custom (OpenAI-compatible)",
            CloudSttApi::OpenAi,
            "http://localhost:8000/v1",
            "whisper-1",
            &[],
            Vocab::Prompt,
        )
    },
];

pub fn provider(id: &str) -> Option<&'static CloudSttProvider> {
    PROVIDERS.iter().find(|p| p.id == id)
}

/// The cloud provider the settings select, or None for local transcription.
pub fn active_provider(settings: &AppSettings) -> Option<&'static CloudSttProvider> {
    provider(&settings.transcription_provider_id)
}

pub fn uses_cloud(settings: &AppSettings) -> bool {
    active_provider(settings).is_some()
}

/// The language hint cloud APIs get: None for auto, else the ISO-639-1 base
/// (`zh-Hans` → `zh`, `en-US` → `en`).
pub fn request_language(settings: &AppSettings) -> Option<String> {
    match settings.selected_language.as_str() {
        "" | "auto" => None,
        lang => Some(base_language(lang).to_string()),
    }
}

/// Everything one request needs, resolved from settings.
struct Request {
    provider: &'static CloudSttProvider,
    base_url: String,
    api_key: String,
    model: String,
    language: Option<String>,
    vocabulary: Vec<String>,
}

impl Request {
    fn from_settings(provider: &'static CloudSttProvider, settings: &AppSettings) -> Result<Self> {
        let api_key = settings
            .transcription_api_keys
            .get(provider.id)
            .map(|k| k.trim().to_string())
            .unwrap_or_default();
        if api_key.is_empty() && provider.requires_api_key {
            return Err(anyhow!(
                "No API key set for {}. Add one in Settings → General → Transcription.",
                provider.label
            ));
        }
        let configured = |map: &std::collections::HashMap<String, String>| {
            map.get(provider.id)
                .map(|v| v.trim().to_string())
                .filter(|v| !v.is_empty())
        };
        let model = configured(&settings.transcription_models)
            .unwrap_or_else(|| provider.default_model.to_string());
        let base_url = configured(&settings.transcription_base_urls)
            .filter(|_| provider.allow_base_url_edit)
            .unwrap_or_else(|| provider.base_url.to_string());
        if base_url.is_empty() {
            return Err(anyhow!(
                "Set the Base URL for {} in Settings → General → Transcription.",
                provider.label
            ));
        }
        Ok(Self {
            base_url: base_url.trim_end_matches('/').to_string(),
            api_key,
            model,
            language: request_language(settings),
            vocabulary: settings.custom_words.clone(),
            provider,
        })
    }

    fn post(&self, path: &str) -> RequestBuilder {
        CLIENT.post(format!("{}{}", self.base_url, path))
    }
}

/// Ok when the selected cloud provider has what it needs to run, so a
/// missing key is reported before the user speaks rather than after.
pub fn check_ready(settings: &AppSettings) -> Result<()> {
    match active_provider(settings) {
        Some(provider) => Request::from_settings(provider, settings).map(|_| ()),
        None => Ok(()),
    }
}

/// Open a pooled connection to the provider while the user is still
/// speaking, so the upload after key release skips connection setup.
pub fn prewarm(settings: &AppSettings) {
    let Some(provider) = active_provider(settings) else {
        return;
    };
    let Ok(req) = Request::from_settings(provider, settings) else {
        return;
    };
    std::thread::spawn(move || {
        // Any response (even 404) leaves a warm keep-alive connection.
        let _ = CLIENT.head(&req.base_url).timeout(PREWARM_TIMEOUT).send();
    });
}

/// Transcribe 16 kHz mono f32 PCM with the selected cloud provider.
///
/// Runs on its own thread: callers may be on an async runtime worker, where
/// the blocking client must not be used.
pub fn transcribe(settings: &AppSettings, samples: &[f32]) -> Result<String> {
    let provider =
        active_provider(settings).ok_or_else(|| anyhow!("No cloud provider selected"))?;
    let request = Request::from_settings(provider, settings)?;
    let wav = crate::audio_toolkit::encode_wav_bytes(samples)?;
    let started = Instant::now();
    let model = request.model.clone();

    let text = std::thread::spawn(move || send(&request, wav))
        .join()
        .map_err(|_| anyhow!("Cloud transcription thread panicked"))??;

    info!(
        "Cloud transcription via {} ({}) took {:.2}s",
        provider.label,
        model,
        started.elapsed().as_secs_f64()
    );
    Ok(text.trim().to_string())
}

fn send(req: &Request, wav: Vec<u8>) -> Result<String> {
    debug!(
        "Cloud STT request: provider={} model={} bytes={}",
        req.provider.id,
        req.model,
        wav.len()
    );
    match req.provider.api {
        CloudSttApi::OpenAi => send_openai(req, wav),
        CloudSttApi::Deepgram => send_deepgram(req, wav),
        CloudSttApi::ElevenLabs => send_elevenlabs(req, wav),
        CloudSttApi::Gemini => send_gemini(req, wav),
        CloudSttApi::Xai => send_xai(req, wav),
        CloudSttApi::AssemblyAi => send_assemblyai(req, wav),
        CloudSttApi::AzureSpeech => send_azure(req, wav),
        CloudSttApi::OpenRouter => send_openrouter(req, wav),
    }
}

fn wav_part(wav: Vec<u8>) -> Result<multipart::Part> {
    multipart::Part::bytes(wav)
        .file_name("audio.wav")
        .mime_str("audio/wav")
        .context("Invalid audio MIME type")
}

/// Add custom words to a multipart form per the provider's [`Vocab`].
fn with_vocab(mut form: multipart::Form, req: &Request) -> multipart::Form {
    if req.vocabulary.is_empty() {
        return form;
    }
    match req.provider.vocab {
        Vocab::Prompt => form.text(
            "prompt",
            format!("Vocabulary: {}.", req.vocabulary.join(", ")),
        ),
        Vocab::Repeated(field) => {
            for word in &req.vocabulary {
                form = form.text(field, word.clone());
            }
            form
        }
        Vocab::Native => form,
    }
}

fn base64(bytes: &[u8]) -> String {
    base64::engine::general_purpose::STANDARD.encode(bytes)
}

/// Concatenate the `text` of each element of a JSON array.
fn join_texts(items: &[Value], separator: &str) -> String {
    items
        .iter()
        .filter_map(|p| p.get("text").and_then(Value::as_str))
        .collect::<Vec<_>>()
        .join(separator)
}

fn send_openai(req: &Request, wav: Vec<u8>) -> Result<String> {
    let mut form = multipart::Form::new()
        .part("file", wav_part(wav)?)
        .text("model", req.model.clone())
        .text("response_format", "json");
    // gpt-transcribe takes array hints instead of `language` and `prompt`.
    if req.model.starts_with("gpt-transcribe") {
        if let Some(lang) = &req.language {
            form = form.text("languages[]", lang.clone());
        }
        for word in &req.vocabulary {
            form = form.text("keywords[]", word.clone());
        }
    } else {
        if let Some(lang) = &req.language {
            form = form.text("language", lang.clone());
        }
        form = with_vocab(form, req);
    }
    let mut builder = req.post("/audio/transcriptions").multipart(form);
    if !req.api_key.is_empty() {
        builder = builder.bearer_auth(&req.api_key);
    }
    string_at(&read_json(builder)?, "/text")
}

fn send_deepgram(req: &Request, wav: Vec<u8>) -> Result<String> {
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
    let builder = req
        .post("/listen")
        .query(&query)
        .header("Authorization", format!("Token {}", req.api_key))
        .header("Content-Type", "audio/wav")
        .body(wav);
    string_at(
        &read_json(builder)?,
        "/results/channels/0/alternatives/0/transcript",
    )
}

fn send_elevenlabs(req: &Request, wav: Vec<u8>) -> Result<String> {
    let mut form = multipart::Form::new()
        .part("file", wav_part(wav)?)
        .text("model_id", req.model.clone())
        .text("tag_audio_events", "false");
    if let Some(lang) = &req.language {
        form = form.text("language_code", lang.clone());
    }
    // Keyterm prompting is a Scribe v2+ feature.
    if req.model != "scribe_v1" {
        form = with_vocab(form, req);
    }
    let builder = req
        .post("/speech-to-text")
        .header("xi-api-key", &req.api_key)
        .multipart(form);
    string_at(&read_json(builder)?, "/text")
}

fn send_gemini(req: &Request, wav: Vec<u8>) -> Result<String> {
    let audio = json!({ "inline_data": { "mime_type": "audio/wav", "data": base64(&wav) } });
    let payload = if req.model.contains("transcribe") {
        // Dedicated transcription model: configured, not prompted.
        let mut config = json!({});
        if let Some(lang) = &req.language {
            config["languageCodes"] = json!([lang]);
        }
        if !req.vocabulary.is_empty() {
            config["customVocabulary"] = json!(req.vocabulary);
        }
        json!({
            "contents": [{ "parts": [audio] }],
            "generationConfig": { "audioTranscriptionConfig": config }
        })
    } else {
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
        json!({
            "contents": [{ "parts": [{ "text": prompt }, audio] }],
            "generationConfig": { "temperature": 0 }
        })
    };
    let builder = req
        .post(&format!("/models/{}:generateContent", req.model))
        .header("x-goog-api-key", &req.api_key)
        .json(&payload);
    let body = read_json(builder)?;
    // A response may split text across parts.
    let parts = body
        .pointer("/candidates/0/content/parts")
        .and_then(Value::as_array)
        .ok_or_else(|| unexpected_shape(&body))?;
    Ok(join_texts(parts, ""))
}

fn send_xai(req: &Request, wav: Vec<u8>) -> Result<String> {
    let mut form = multipart::Form::new().text("audio_format", "wav");
    if let Some(lang) = &req.language {
        // Formatting (numbers, punctuation normalisation) requires a language.
        form = form.text("language", lang.clone()).text("format", "true");
    }
    // The file must be the last field.
    form = with_vocab(form, req).part("file", wav_part(wav)?);
    let builder = req.post("/stt").bearer_auth(&req.api_key).multipart(form);
    string_at(&read_json(builder)?, "/text")
}

fn send_assemblyai(req: &Request, wav: Vec<u8>) -> Result<String> {
    let auth = |b: RequestBuilder| b.header("authorization", &req.api_key);
    let upload = read_json(
        auth(req.post("/upload"))
            .header("Content-Type", "application/octet-stream")
            .body(wav),
    )?;
    let audio_url = string_at(&upload, "/upload_url")?;

    let mut payload = json!({ "audio_url": audio_url, "speech_models": [req.model] });
    match &req.language {
        Some(lang) => payload["language_code"] = json!(lang),
        None => payload["language_detection"] = json!(true),
    }
    if !req.vocabulary.is_empty() {
        payload["keyterms_prompt"] = json!(req.vocabulary);
    }
    let created = read_json(auth(req.post("/transcript")).json(&payload))?;
    let job_url = format!(
        "{}/transcript/{}",
        req.base_url,
        string_at(&created, "/id")?
    );

    let deadline = Instant::now() + POLL_TIMEOUT;
    loop {
        let job = read_json(auth(CLIENT.get(&job_url)))?;
        match job.get("status").and_then(Value::as_str) {
            Some("completed") => return Ok(string_at(&job, "/text").unwrap_or_default()),
            Some("error") => {
                return Err(anyhow!(
                    "AssemblyAI transcription failed: {}",
                    job.get("error")
                        .and_then(Value::as_str)
                        .unwrap_or("unknown")
                ))
            }
            _ if Instant::now() > deadline => {
                return Err(anyhow!("AssemblyAI transcription timed out"))
            }
            _ => std::thread::sleep(POLL_INTERVAL),
        }
    }
}

fn send_azure(req: &Request, wav: Vec<u8>) -> Result<String> {
    let mut definition = json!({ "enhancedMode": { "enabled": true, "model": req.model } });
    if let Some(lang) = &req.language {
        definition["locales"] = json!([lang]);
    }
    if !req.vocabulary.is_empty() {
        definition["phraseList"] = json!({ "phrases": req.vocabulary });
    }
    let form = multipart::Form::new()
        .part("audio", wav_part(wav)?)
        .text("definition", definition.to_string());
    let builder = req
        .post("/speechtotext/transcriptions:transcribe?api-version=2025-10-15")
        .header("Ocp-Apim-Subscription-Key", &req.api_key)
        .multipart(form);
    let body = read_json(builder)?;
    if let Ok(text) = string_at(&body, "/combinedPhrases/0/text") {
        return Ok(text);
    }
    let phrases = body
        .get("phrases")
        .and_then(Value::as_array)
        .ok_or_else(|| unexpected_shape(&body))?;
    Ok(join_texts(phrases, " "))
}

fn send_openrouter(req: &Request, wav: Vec<u8>) -> Result<String> {
    let mut payload = json!({
        "model": req.model,
        "input_audio": { "data": base64(&wav), "format": "wav" },
    });
    if let Some(lang) = &req.language {
        payload["language"] = json!(lang);
    }
    let builder = req
        .post("/audio/transcriptions")
        .bearer_auth(&req.api_key)
        .json(&payload);
    string_at(&read_json(builder)?, "/text")
}

fn read_json(builder: RequestBuilder) -> Result<Value> {
    let response = builder.send().map_err(|e| {
        let message =
            crate::llm_client::report_reqwest_error("Cloud transcription request failed", &e);
        error!("{}", message);
        anyhow!(message)
    })?;
    let status = response.status();
    let text = response
        .text()
        .context("Failed to read cloud transcription response")?;
    if !status.is_success() {
        // Error bodies carry the provider's reason (bad key, unknown model).
        return Err(anyhow!(
            "Cloud transcription failed ({}): {}",
            status,
            truncate(&text)
        ));
    }
    serde_json::from_str(&text).map_err(|_| {
        anyhow!(
            "Cloud transcription returned non-JSON: {}",
            crate::utils::redact_text(&truncate(&text))
        )
    })
}

fn unexpected_shape(body: &Value) -> anyhow::Error {
    // A success body may hold transcript text, so redact it in release builds.
    anyhow!(
        "Unexpected cloud transcription response: {}",
        crate::utils::redact_text(&truncate(&body.to_string()))
    )
}

/// The string at a JSON pointer (`/a/0/b`).
fn string_at(body: &Value, pointer: &str) -> Result<String> {
    body.pointer(pointer)
        .and_then(Value::as_str)
        .map(str::to_string)
        .ok_or_else(|| unexpected_shape(body))
}

fn truncate(s: &str) -> String {
    const MAX: usize = 500;
    if s.len() <= MAX {
        return s.to_string();
    }
    let mut end = MAX;
    while !s.is_char_boundary(end) {
        end -= 1;
    }
    format!("{}…", &s[..end])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn provider_ids_are_unique_and_not_local() {
        let mut ids: Vec<_> = PROVIDERS.iter().map(|p| p.id).collect();
        let total = ids.len();
        ids.sort();
        ids.dedup();
        assert_eq!(ids.len(), total);
        assert!(provider(LOCAL_PROVIDER_ID).is_none());
    }

    #[test]
    fn providers_without_fixed_base_url_allow_editing() {
        for p in PROVIDERS {
            assert!(!p.base_url.is_empty() || p.allow_base_url_edit, "{}", p.id);
        }
    }

    #[test]
    fn string_at_follows_pointer() {
        let body = json!({"results":{"channels":[{"alternatives":[{"transcript":"hi"}]}]}});
        assert_eq!(
            string_at(&body, "/results/channels/0/alternatives/0/transcript").unwrap(),
            "hi"
        );
    }
}
