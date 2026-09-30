# RenderDoc in-application API header

`renderdoc_app.h`, vendored unmodified from
<https://github.com/baldurk/renderdoc> tag `v1.41`
(`renderdoc/api/app/renderdoc_app.h`, SHA-256
`e090464ca63e63a1a087e30875a0447aad30eadae4d5bff75b6731cd1643c310`).
MIT licensed; the license text is in the header.

The engine never loads RenderDoc. When a trace requests
`TraceOptions.external.capture` and RenderDoc has already injected itself
into the process, the engine finds it with `RTLD_NOLOAD` and brackets the
capture with `StartFrameCapture`/`EndFrameCapture`, because headless compute
has no present call to delimit a frame.
