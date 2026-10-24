//! What the user is dictating into: the frontmost app, its focused window,
//! and the text around the cursor. Captured when recording starts so
//! post-processing can pick a per-app prompt and give the LLM context, and so
//! the dot overlay can sit beside the text field.
//!
//! Only macOS has a real implementation (Accessibility API); elsewhere the
//! context is empty and every feature built on it degrades to its default.

use log::debug;
use serde::Serialize;
use specta::Type;
use std::sync::{Condvar, Mutex};
use std::time::Duration;

#[derive(Debug, Clone, Default, PartialEq)]
pub struct DictationContext {
    /// Bundle identifier of the frontmost app, e.g. `com.tinyspeck.slackmacgap`.
    pub bundle_id: Option<String>,
    pub app_name: Option<String>,
    pub window_title: Option<String>,
    /// Text selected in the focused field (dictation will replace it).
    pub selected_text: Option<String>,
    /// Text in the focused field before the cursor.
    pub text_before_cursor: Option<String>,
    /// Where the dot overlay should be centred, in logical screen points
    /// (top-left origin): just past the caret, or beside the focused field.
    pub anchor: Option<(f64, f64)>,
}

/// A running app the user can attach a per-app prompt to.
#[derive(Debug, Clone, Serialize, Type, PartialEq)]
pub struct RunningApp {
    pub bundle_id: String,
    pub name: String,
}

/// What a capture should collect beyond the app identity.
#[derive(Debug, Clone, Copy)]
#[cfg_attr(not(target_os = "macos"), allow(dead_code))]
pub struct CaptureOptions {
    /// Read window title and field text (only sent to the LLM when
    /// context awareness is on).
    pub include_text: bool,
    /// Locate the caret or field for the dot overlay.
    pub include_anchor: bool,
}

struct Slot {
    session: u64,
    context: Option<DictationContext>,
}

static SLOT: Mutex<Slot> = Mutex::new(Slot {
    session: 0,
    context: None,
});
static SLOT_READY: Condvar = Condvar::new();

/// Longest a transcription waits for a capture still in flight. Accessibility
/// calls are capped well below this, so it only bites when the target app is
/// hung.
const TAKE_TIMEOUT: Duration = Duration::from_millis(400);

/// Captures the context for recording `session` on a background thread (an
/// unresponsive app can stall Accessibility calls) and runs `on_ready` with
/// the result. The result is also kept for [`take`].
pub fn begin_capture<F>(session: u64, options: CaptureOptions, on_ready: F)
where
    F: FnOnce(&DictationContext) + Send + 'static,
{
    {
        let mut slot = SLOT.lock().unwrap_or_else(|e| e.into_inner());
        slot.session = session;
        slot.context = None;
    }

    let spawned = std::thread::Builder::new()
        .name("dictation-context".into())
        .spawn(move || {
            let started = std::time::Instant::now();
            let context = capture(options);
            debug!(
                "Captured dictation context in {:?}: app={:?} window={} field_chars={} selection_chars={} anchor={:?}",
                started.elapsed(),
                context.bundle_id,
                context.window_title.is_some(),
                context.text_before_cursor.as_ref().map_or(0, |t| t.chars().count()),
                context.selected_text.as_ref().map_or(0, |t| t.chars().count()),
                context.anchor,
            );
            on_ready(&context);

            let mut slot = SLOT.lock().unwrap_or_else(|e| e.into_inner());
            if slot.session == session {
                slot.context = Some(context);
                SLOT_READY.notify_all();
            }
        });
    if let Err(e) = spawned {
        log::warn!("Failed to start dictation context capture: {e}");
    }
}

/// Takes the context captured for `session`, waiting briefly if the capture
/// is still running. `None` if it belongs to another session or timed out.
pub fn take(session: u64) -> Option<DictationContext> {
    let slot = SLOT.lock().unwrap_or_else(|e| e.into_inner());
    let (mut slot, _) = SLOT_READY
        .wait_timeout_while(slot, TAKE_TIMEOUT, |s| {
            s.session == session && s.context.is_none()
        })
        .unwrap_or_else(|e| e.into_inner());
    if slot.session != session {
        return None;
    }
    slot.context.take()
}

#[cfg(target_os = "macos")]
pub fn capture(options: CaptureOptions) -> DictationContext {
    macos::capture(options)
}

#[cfg(not(target_os = "macos"))]
pub fn capture(_options: CaptureOptions) -> DictationContext {
    DictationContext::default()
}

/// Regular (Dock-visible) apps currently running, for the per-app prompt picker.
#[cfg(target_os = "macos")]
pub fn running_apps() -> Vec<RunningApp> {
    macos::running_apps()
}

#[cfg(not(target_os = "macos"))]
pub fn running_apps() -> Vec<RunningApp> {
    Vec::new()
}

#[cfg(target_os = "macos")]
mod macos {
    use super::{CaptureOptions, DictationContext, RunningApp};
    use core_foundation::base::{CFGetTypeID, CFRelease, CFTypeID, CFTypeRef, TCFType};
    use core_foundation::string::{CFString, CFStringRef};
    use objc2_app_kit::{NSApplicationActivationPolicy, NSRunningApplication, NSWorkspace};
    use std::ffi::c_void;

    type AXUIElementRef = *const c_void;
    type AXValueRef = *const c_void;
    type AXError = i32;

    const AX_SUCCESS: AXError = 0;
    // AXValueType
    const AX_VALUE_CG_POINT: u32 = 1;
    const AX_VALUE_CG_SIZE: u32 = 2;
    const AX_VALUE_CG_RECT: u32 = 3;
    const AX_VALUE_CF_RANGE: u32 = 4;

    /// Per-call cap so a hung target app can't stall the capture.
    const AX_TIMEOUT_SECONDS: f32 = 0.15;
    /// Longest field value read; long documents are cut to the text nearest
    /// the cursor later, but very large values aren't worth copying at all.
    const MAX_FIELD_UTF16: usize = 200_000;
    /// Gap between the caret or field edge and the dot's centre.
    const DOT_GAP: f64 = 14.0;

    #[repr(C)]
    #[derive(Debug, Clone, Copy, Default)]
    struct CGPoint {
        x: f64,
        y: f64,
    }

    #[repr(C)]
    #[derive(Debug, Clone, Copy, Default)]
    struct CGSize {
        width: f64,
        height: f64,
    }

    #[repr(C)]
    #[derive(Debug, Clone, Copy, Default)]
    struct CGRect {
        origin: CGPoint,
        size: CGSize,
    }

    #[repr(C)]
    #[derive(Debug, Clone, Copy, Default)]
    struct CFRange {
        location: isize,
        length: isize,
    }

    #[link(name = "ApplicationServices", kind = "framework")]
    extern "C" {
        fn AXIsProcessTrusted() -> bool;
        fn AXUIElementCreateSystemWide() -> AXUIElementRef;
        fn AXUIElementCopyAttributeValue(
            element: AXUIElementRef,
            attribute: CFStringRef,
            value: *mut CFTypeRef,
        ) -> AXError;
        fn AXUIElementCopyParameterizedAttributeValue(
            element: AXUIElementRef,
            attribute: CFStringRef,
            parameter: CFTypeRef,
            value: *mut CFTypeRef,
        ) -> AXError;
        fn AXUIElementGetPid(element: AXUIElementRef, pid: *mut i32) -> AXError;
        fn AXUIElementSetMessagingTimeout(element: AXUIElementRef, seconds: f32) -> AXError;
        fn AXValueGetTypeID() -> CFTypeID;
        fn AXValueGetType(value: AXValueRef) -> u32;
        fn AXValueGetValue(value: AXValueRef, value_type: u32, out: *mut c_void) -> bool;
        fn AXValueCreate(value_type: u32, value: *const c_void) -> AXValueRef;
    }

    /// An owned CoreFoundation reference (released on drop).
    struct Cf(CFTypeRef);

    impl Drop for Cf {
        fn drop(&mut self) {
            if !self.0.is_null() {
                unsafe { CFRelease(self.0) };
            }
        }
    }

    impl Cf {
        fn as_element(&self) -> AXUIElementRef {
            self.0 as AXUIElementRef
        }

        fn to_string(&self) -> Option<String> {
            unsafe {
                if CFGetTypeID(self.0) != CFString::type_id() {
                    return None;
                }
                let s = CFString::wrap_under_get_rule(self.0 as CFStringRef).to_string();
                Some(s)
            }
        }

        fn ax_value<T: Default>(&self, value_type: u32) -> Option<T> {
            unsafe {
                if CFGetTypeID(self.0) != AXValueGetTypeID()
                    || AXValueGetType(self.0 as AXValueRef) != value_type
                {
                    return None;
                }
                let mut out = T::default();
                AXValueGetValue(
                    self.0 as AXValueRef,
                    value_type,
                    &mut out as *mut T as *mut c_void,
                )
                .then_some(out)
            }
        }
    }

    fn attr(element: AXUIElementRef, name: &str) -> Option<Cf> {
        let name = CFString::new(name);
        let mut value: CFTypeRef = std::ptr::null();
        let err = unsafe {
            AXUIElementCopyAttributeValue(element, name.as_concrete_TypeRef(), &mut value)
        };
        (err == AX_SUCCESS && !value.is_null()).then_some(Cf(value))
    }

    fn string_attr(element: AXUIElementRef, name: &str) -> Option<String> {
        attr(element, name)?
            .to_string()
            .filter(|s| !s.trim().is_empty())
    }

    fn bounds_for_range(element: AXUIElementRef, range: CFRange) -> Option<CGRect> {
        let param =
            unsafe { AXValueCreate(AX_VALUE_CF_RANGE, &range as *const CFRange as *const c_void) };
        if param.is_null() {
            return None;
        }
        let param = Cf(param as CFTypeRef);
        let name = CFString::new("AXBoundsForRange");
        let mut value: CFTypeRef = std::ptr::null();
        let err = unsafe {
            AXUIElementCopyParameterizedAttributeValue(
                element,
                name.as_concrete_TypeRef(),
                param.0,
                &mut value,
            )
        };
        if err != AX_SUCCESS || value.is_null() {
            return None;
        }
        Cf(value)
            .ax_value::<CGRect>(AX_VALUE_CG_RECT)
            .filter(|r| r.size.height > 0.0 && (r.origin.x != 0.0 || r.origin.y != 0.0))
    }

    fn element_frame(element: AXUIElementRef) -> Option<CGRect> {
        let origin = attr(element, "AXPosition")?.ax_value::<CGPoint>(AX_VALUE_CG_POINT)?;
        let size = attr(element, "AXSize")?.ax_value::<CGSize>(AX_VALUE_CG_SIZE)?;
        (size.width > 0.0 && size.height > 0.0).then_some(CGRect { origin, size })
    }

    /// Caret position: bounds of the character just before the insertion
    /// point (zero-length ranges often report an empty rect), using its right
    /// edge.
    fn caret_anchor(element: AXUIElementRef, selection: CFRange) -> Option<(f64, f64)> {
        let (range, use_right_edge) = if selection.length > 0 {
            (selection, true)
        } else if selection.location > 0 {
            (
                CFRange {
                    location: selection.location - 1,
                    length: 1,
                },
                true,
            )
        } else {
            (selection, false)
        };
        let rect = bounds_for_range(element, range)?;
        let x = if use_right_edge {
            rect.origin.x + rect.size.width
        } else {
            rect.origin.x
        };
        Some((x + DOT_GAP, rect.origin.y + rect.size.height / 2.0))
    }

    /// Beside a field: past its right edge for single-line fields, inside the
    /// top-right corner for tall ones (a chat box or document).
    fn field_anchor(frame: CGRect) -> (f64, f64) {
        if frame.size.height <= 60.0 {
            (
                frame.origin.x + frame.size.width + DOT_GAP,
                frame.origin.y + frame.size.height / 2.0,
            )
        } else {
            (
                frame.origin.x + frame.size.width - DOT_GAP,
                frame.origin.y + DOT_GAP,
            )
        }
    }

    fn utf16_prefix(text: &str, utf16_len: usize) -> String {
        let units: Vec<u16> = text.encode_utf16().take(utf16_len).collect();
        String::from_utf16_lossy(&units)
    }

    fn utf16_slice(text: &str, start: usize, len: usize) -> String {
        let units: Vec<u16> = text.encode_utf16().skip(start).take(len).collect();
        String::from_utf16_lossy(&units)
    }

    fn app_identity(pid: i32) -> (Option<String>, Option<String>) {
        let app = NSRunningApplication::runningApplicationWithProcessIdentifier(pid);
        match app {
            Some(app) => (
                app.bundleIdentifier().map(|s| s.to_string()),
                app.localizedName().map(|s| s.to_string()),
            ),
            None => (None, None),
        }
    }

    fn frontmost_identity() -> (Option<String>, Option<String>) {
        let workspace = NSWorkspace::sharedWorkspace();
        match workspace.frontmostApplication() {
            Some(app) => (
                app.bundleIdentifier().map(|s| s.to_string()),
                app.localizedName().map(|s| s.to_string()),
            ),
            None => (None, None),
        }
    }

    pub fn capture(options: CaptureOptions) -> DictationContext {
        let mut ctx = DictationContext::default();

        if !unsafe { AXIsProcessTrusted() } {
            let (bundle_id, app_name) = frontmost_identity();
            ctx.bundle_id = bundle_id;
            ctx.app_name = app_name;
            return ctx;
        }

        let system = Cf(unsafe { AXUIElementCreateSystemWide() } as CFTypeRef);
        if system.0.is_null() {
            return ctx;
        }
        unsafe { AXUIElementSetMessagingTimeout(system.as_element(), AX_TIMEOUT_SECONDS) };

        let Some(app) = attr(system.as_element(), "AXFocusedApplication") else {
            let (bundle_id, app_name) = frontmost_identity();
            ctx.bundle_id = bundle_id;
            ctx.app_name = app_name;
            return ctx;
        };
        unsafe { AXUIElementSetMessagingTimeout(app.as_element(), AX_TIMEOUT_SECONDS) };

        let mut pid = 0;
        if unsafe { AXUIElementGetPid(app.as_element(), &mut pid) } == AX_SUCCESS {
            let (bundle_id, app_name) = app_identity(pid);
            ctx.bundle_id = bundle_id;
            ctx.app_name = app_name;
        }

        if !options.include_text && !options.include_anchor {
            return ctx;
        }

        if options.include_text {
            if let Some(window) = attr(app.as_element(), "AXFocusedWindow") {
                ctx.window_title = string_attr(window.as_element(), "AXTitle");
            }
        }

        let Some(focused) = attr(app.as_element(), "AXFocusedUIElement") else {
            return ctx;
        };
        let element = focused.as_element();

        // Never read password fields.
        let is_secure = ["AXRole", "AXSubrole"]
            .iter()
            .any(|a| string_attr(element, a).as_deref() == Some("AXSecureTextField"));

        let selection = attr(element, "AXSelectedTextRange")
            .and_then(|v| v.ax_value::<CFRange>(AX_VALUE_CF_RANGE))
            .filter(|r| r.location >= 0 && r.length >= 0);

        if options.include_text && !is_secure {
            ctx.selected_text = string_attr(element, "AXSelectedText");
            if let Some(value) = attr(element, "AXValue").and_then(|v| v.to_string()) {
                let utf16_len = value.encode_utf16().count();
                if utf16_len <= MAX_FIELD_UTF16 {
                    let before = match selection {
                        Some(r) => utf16_prefix(&value, r.location as usize),
                        None => value.clone(),
                    };
                    if !before.trim().is_empty() {
                        ctx.text_before_cursor = Some(before);
                    }
                    if ctx.selected_text.is_none() {
                        if let Some(r) = selection.filter(|r| r.length > 0) {
                            ctx.selected_text =
                                Some(utf16_slice(&value, r.location as usize, r.length as usize))
                                    .filter(|s| !s.trim().is_empty());
                        }
                    }
                }
            }
        }

        if options.include_anchor {
            ctx.anchor = selection
                .and_then(|r| caret_anchor(element, r))
                .or_else(|| element_frame(element).map(field_anchor));
        }

        ctx
    }

    pub fn running_apps() -> Vec<RunningApp> {
        let workspace = NSWorkspace::sharedWorkspace();
        let mut apps: Vec<RunningApp> = workspace
            .runningApplications()
            .iter()
            .filter(|app| app.activationPolicy() == NSApplicationActivationPolicy::Regular)
            .filter_map(|app| {
                Some(RunningApp {
                    bundle_id: app.bundleIdentifier()?.to_string(),
                    name: app.localizedName()?.to_string(),
                })
            })
            .collect();
        apps.sort_by_key(|a| a.name.to_lowercase());
        apps.dedup_by(|a, b| a.bundle_id == b.bundle_id);
        apps
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn take_returns_only_the_matching_session() {
        let opts = CaptureOptions {
            include_text: false,
            include_anchor: false,
        };
        begin_capture(41, opts, |_| {});
        assert!(take(41).is_some());
        // Taken once; a second take times out empty.
        assert!(take(41).is_none());

        begin_capture(42, opts, |_| {});
        assert!(take(7).is_none());
    }
}
