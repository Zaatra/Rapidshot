//! Whether this thread can see the desktop it is trying to capture.
//!
//! The Rust half of `rapidshot/util/desktop.py`, with the same messages.
//! `DuplicateOutput` reports both "protected content is on screen" and "you are
//! not on the input desktop" as a bare `E_ACCESSDENIED`; this is what tells the
//! two apart, so a locked workstation is not reported as a DRM player window.
//!
//! It also follows the input desktop (`attach_thread_to_input_desktop`), which
//! is how a SYSTEM process in the user's session captures the lock screen, the
//! sign-in box, a UAC prompt and Ctrl+Alt+Del: all of them are the secure
//! desktop (`Winlogon`), and only SYSTEM may open it.

use std::cell::Cell;

use windows::Win32::Foundation::HANDLE;
use windows::Win32::System::StationsAndDesktops::{
    CloseDesktop, GetThreadDesktop, GetUserObjectInformationW, OpenInputDesktop, SetThreadDesktop,
    DESKTOP_ACCESS_FLAGS, DESKTOP_CONTROL_FLAGS, HDESK, UOI_NAME,
};
use windows::Win32::System::Threading::GetCurrentThreadId;

const GENERIC_READ: u32 = 0x8000_0000;
const GENERIC_ALL: u32 = 0x1000_0000;

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

/// The desktop this thread is attached to, and so the one it captures.
pub fn thread_desktop_name() -> Option<String> {
    unsafe { GetThreadDesktop(GetCurrentThreadId()) }
        .ok()
        .and_then(desktop_name)
}

/// The desktop receiving input, or `None` when this process may not open it:
/// the secure desktop, to anything but SYSTEM.
pub fn input_desktop_name() -> Option<String> {
    let input = unsafe {
        OpenInputDesktop(
            DESKTOP_CONTROL_FLAGS(0),
            false,
            DESKTOP_ACCESS_FLAGS(GENERIC_READ),
        )
    }
    .ok()?;
    let name = desktop_name(input);
    let _ = unsafe { CloseDesktop(input) };
    name
}

/// The input desktop is not the one this thread captures. That is what lock,
/// sign-in, UAC and Ctrl+Alt+Del look like from a capture thread: an
/// expected event to follow (or wait out), not a fault to report.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DesktopSwitch {
    /// The desktop this thread is attached to.
    pub thread_desktop: Option<String>,
    /// The desktop receiving input; `None` when this process cannot open it.
    pub input_desktop: Option<String>,
}

impl std::fmt::Display for DesktopSwitch {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let from = self.thread_desktop.as_deref().unwrap_or("?");
        match &self.input_desktop {
            Some(to) => write!(f, "the input desktop is '{to}', this thread is on '{from}'"),
            None => write!(
                f,
                "the input desktop is the secure one (lock, sign-in, UAC or \
                 Ctrl+Alt+Del), which only SYSTEM can open; this thread is on '{from}'"
            ),
        }
    }
}

/// `Some` when the input desktop is not this thread's. `None` when they match
/// or either name cannot be read: a switch is only reported when it is seen.
pub fn desktop_switch() -> Option<DesktopSwitch> {
    let state = describe_desktop_access();
    if state.is_input_desktop != Some(false) {
        return None;
    }
    Some(DesktopSwitch {
        thread_desktop: state.thread_desktop,
        input_desktop: state.input_desktop,
    })
}

thread_local! {
    /// The desktop handle this module attached the thread to. A thread's
    /// desktop handle must stay open while it is attached, and the previous
    /// one is closed once the thread has moved off it.
    static ATTACHED: Cell<Option<isize>> = const { Cell::new(None) };
}

/// Move the calling thread onto the input desktop if it is not already there.
/// `Ok(Some(name))` when it moved, `Ok(None)` when it was there already.
///
/// `SetThreadDesktop` refuses a thread that owns a window or a hook, so call
/// this from a thread that does neither: a capture thread, or an input thread
/// that only calls `SendInput`. Opening the secure desktop needs SYSTEM; for
/// anybody else that is the error, and capture has to wait for the user's
/// desktop to come back.
pub fn attach_thread_to_input_desktop() -> Result<Option<String>, String> {
    let input = unsafe {
        OpenInputDesktop(
            DESKTOP_CONTROL_FLAGS(0),
            false,
            DESKTOP_ACCESS_FLAGS(GENERIC_ALL),
        )
    }
    .map_err(|e| {
        format!(
            "OpenInputDesktop: {e} (the secure desktop opens only to SYSTEM, \
             so this is the lock, sign-in, UAC or Ctrl+Alt+Del screen)"
        )
    })?;
    let target = desktop_name(input);
    if target.is_some() && target == thread_desktop_name() {
        let _ = unsafe { CloseDesktop(input) };
        return Ok(None);
    }
    if let Err(e) = unsafe { SetThreadDesktop(input) } {
        let _ = unsafe { CloseDesktop(input) };
        return Err(format!(
            "SetThreadDesktop: {e} (a thread that owns a window or a hook cannot move)"
        ));
    }
    if let Some(previous) = ATTACHED.with(|a| a.replace(Some(input.0 as isize))) {
        let _ = unsafe { CloseDesktop(HDESK(previous as *mut std::ffi::c_void)) };
    }
    Ok(Some(target.unwrap_or_else(|| "?".to_owned())))
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU32, Ordering};
    use windows::core::HSTRING;
    use windows::Win32::System::StationsAndDesktops::CreateDesktopW;

    /// Run `f` on a thread of its own, attached to a fresh desktop that is not
    /// the input desktop: the position a capture thread is in once the lock
    /// screen or a UAC prompt has taken over. Any user may create one.
    pub(crate) fn on_another_desktop<T: Send + 'static>(
        f: impl FnOnce(&str) -> T + Send + 'static,
    ) -> T {
        static N: AtomicU32 = AtomicU32::new(0);
        let name = format!(
            "rapidshot-test-{}-{}",
            std::process::id(),
            N.fetch_add(1, Ordering::Relaxed)
        );
        std::thread::spawn(move || {
            let desk = unsafe {
                CreateDesktopW(
                    &HSTRING::from(name.as_str()),
                    None,
                    None,
                    DESKTOP_CONTROL_FLAGS(0),
                    GENERIC_ALL,
                    None,
                )
            }
            .expect("CreateDesktopW");
            unsafe { SetThreadDesktop(desk) }.expect("SetThreadDesktop to the test desktop");
            f(&name)
        })
        .join()
        .unwrap()
    }

    #[test]
    fn the_input_desktop_is_not_a_switch() {
        assert_eq!(desktop_switch(), None);
        assert_eq!(thread_desktop_name().as_deref(), Some("Default"));
        assert_eq!(input_desktop_name().as_deref(), Some("Default"));
    }

    #[test]
    fn a_thread_on_another_desktop_sees_a_switch_and_can_follow_back() {
        on_another_desktop(|name| {
            assert_eq!(
                desktop_switch(),
                Some(DesktopSwitch {
                    thread_desktop: Some(name.to_owned()),
                    input_desktop: Some("Default".to_owned()),
                })
            );
            assert_eq!(
                attach_thread_to_input_desktop(),
                Ok(Some("Default".to_owned()))
            );
            assert_eq!(thread_desktop_name().as_deref(), Some("Default"));
            assert_eq!(desktop_switch(), None);
            // Already there: nothing moves, nothing is opened and kept.
            assert_eq!(attach_thread_to_input_desktop(), Ok(None));
        });
    }

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
