# Security Policy

## Supported versions

| Version | Supported |
| --- | --- |
| 2.0.x | Yes |
| 1.x | No — please upgrade; see the migration notes in `CHANGELOG.md` |

## Reporting a vulnerability

Report privately through GitHub:
**[Security → Report a vulnerability](https://github.com/Zaatra/Rapidshot/security/advisories/new)**

Please do not open a public issue for a security problem. Private reporting
lets a fix ship before the details are public.

Include what you have — a description, the version, and anything that
reproduces it. A partial report is worth sending; do not wait until you have a
polished one.

**Expect a first response within 7 days.** This is a small project, so that is a
realistic commitment rather than an aspirational one. If you have not heard back
in that time, the report has gone astray — please ping the issue tracker without
details and ask to be contacted.

## What is in scope

Rapidshot captures the screen and hands frames to the calling process. The
security-relevant surface is smaller than that sounds, and these are the parts
worth reporting:

- **Memory safety in the frame path.** Buffer sizing, pitch and region
  arithmetic, and the pooled-buffer lifetime. `shot()` is known to overrun an
  undersized destination — see below.
- **The optional Rust extension.** It handles raw D3D11/D3D12 resource pointers
  supplied by the caller. Anything that turns a valid call into a bad pointer
  dereference is in scope.
- **Protected content.** Rapidshot surfaces HDCP/DRM refusals rather than
  working around them. A way to capture content the OS masked out is in scope.
- **Dependency and supply-chain issues** in what a `pip install rapidshot`
  actually pulls.

## Known issue, already public

`shot()` writes BGRA regardless of `output_color` and does not bounds-check an
undersized destination buffer. It is documented in `ROADMAP.md` § 10 and needs no
new report. Prefer `grab()`, which validates buffer shape.

## What is not a vulnerability

- **Rapidshot captures the screen.** That is the entire library. A program using
  it can see what is on the display, and that is not a flaw in Rapidshot.
- **Capture requires no special privilege.** Desktop Duplication is available to
  any process in the user's session. That is a Windows design decision.
- **A caller reading a released `PooledBuffer`.** It raises
  `BufferReleasedError` by design. A path where it silently returns another
  frame's pixels instead *is* worth reporting.
