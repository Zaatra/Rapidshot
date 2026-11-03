//! Whether this thread can see the desktop it is trying to capture.
//!
//! The Rust half of `rapidshot/util/desktop.py`, with the same messages.
//! `DuplicateOutput` reports both "protected content is on screen" and "you are
//! not on the input desktop" as a bare `E_ACCESSDENIED`; this is what tells the
//! two apart, so a locked workstation is not reported as a DRM player window.

use windows::Win32::Foundation::HANDLE;
use windows::Win32::System::StationsAndDesktops::{
    CloseDesktop, GetThreadDesktop, GetUserObjectInformationW, OpenInputDesktop,
    DESKTOP_ACCESS_FLAGS, DESKTOP_CONTROL_FLAGS, HDESK, UOI_NAME,
};
use windows::Win32::System::Threading::GetCurrentThreadId;

const GENERIC_READ: u32 = 0x8000_0000;

/// What the thread is attached to and what is receiving input.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct DesktopState {
    pub thread_desktop: Option<String>,
    pub input_desktop: Option<String>,
    /// `Some(true)` only on a confirmed match; `None` when either name could
    /// not be read. A diagnostic that guesses is worse than one that abstains.
    pub is_input_desktop: Option<bool>,
    /// One line saying why capture cannot work from here, when it cannot.
    pub blocked_reason: Option<String>,
}

fn desktop_name(handle: HDESK) -> Option<String> {
    if handle.is_invalid() {
        return None;
    }
    let object = HANDLE(handle.0);
    let mut needed = 0u32;
    // The first call fails by design and reports the size it needs.
    let _ = unsafe { GetUserObjectInformationW(object, UOI_NAME, None, 0, Some(&mut needed)) };
    if needed == 0 {
        return None;
    }
    let mut buffer = vec![0u16; (needed as usize).div_ceil(2)];
    unsafe {
        GetUserObjectInformationW(
            object,
            UOI_NAME,
            Some(buffer.as_mut_ptr().cast()),
            needed,
            Some(&mut needed),
        )
    }
    .ok()?;
    let end = buffer.iter().position(|&c| c == 0).unwrap_or(buffer.len());
    Some(String::from_utf16_lossy(&buffer[..end]))
}

/// Never fails: this runs while building an error message, and a diagnostic
/// that can itself fail would replace a misleading message with no message.
pub fn describe_desktop_access() -> DesktopState {
    let thread_name = unsafe { GetThreadDesktop(GetCurrentThreadId()) }
        .ok()
        .and_then(desktop_name);

    // OpenInputDesktop fails outright when the input desktop is the secure one
    // (lock screen, UAC prompt, Ctrl+Alt+Del), which is exactly the case worth
    // naming.
    let input = unsafe {
        OpenInputDesktop(
            DESKTOP_CONTROL_FLAGS(0),
            false,
            DESKTOP_ACCESS_FLAGS(GENERIC_READ),
        )
    };
    let Ok(input) = input else {
        return DesktopState {
            thread_desktop: thread_name,
            input_desktop: None,
            is_input_desktop: Some(false),
            blocked_reason: Some(
                "the input desktop cannot be opened, which means the secure \
                 desktop is active — the workstation is locked, a UAC prompt \
                 is open, or Ctrl+Alt+Del is showing. Desktop Duplication \
                 cannot capture the secure desktop."
                    .to_owned(),
            ),
        };
    };
    let input_name = desktop_name(input);
    let _ = unsafe { CloseDesktop(input) };

    if let (Some(thread), Some(input)) = (&thread_name, &input_name) {
        if thread != input {
            return DesktopState {
                thread_desktop: thread_name.clone(),
                input_desktop: input_name.clone(),
                is_input_desktop: Some(false),
                blocked_reason: Some(format!(
                    "this thread is attached to desktop '{thread}' but '{input}' is \
                     receiving input. Desktop Duplication only works on the input \
                     desktop — a Session 0 service or a thread moved with \
                     SetThreadDesktop cannot capture the user's screen directly."
                )),
            };
        }
    }
    let is_input = match (&thread_name, &input_name) {
        (Some(t), Some(i)) => Some(t == i),
        _ => None,
    };
    DesktopState {
        thread_desktop: thread_name,
        input_desktop: input_name,
        is_input_desktop: is_input,
        blocked_reason: None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_ordinary_session_is_not_blocked() {
        // The test runs on the interactive desktop, so this is the common case;
        // the blocked branches are exercised from Python by moving a thread to
        // a fresh desktop, which is how the Python version was measured.
        let state = describe_desktop_access();
        assert_eq!(state.blocked_reason, None, "{state:?}");
        assert_eq!(state.is_input_desktop, Some(true), "{state:?}");
        assert_eq!(state.thread_desktop.as_deref(), Some("Default"));
    }
}
