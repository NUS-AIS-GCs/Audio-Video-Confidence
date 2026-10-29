"""Local dark-first product styling; no remote fonts, analytics or assets."""
from deployment.ui.result_sections import RESULT_CSS

BRAND_CSS = """
:root, body, .dark, .gradio-container, .dark .gradio-container {
 color-scheme: dark;
 --av-teal:#63e6d4; --av-green:#73d7ff; --av-ink:#edf4fc; --av-muted:#a5b5cc;
 --av-card:#101b2d; --av-line:#2a3b53; --av-bg:#080f1e;
 --av-ok:#63e6b7; --av-warn:#f4cc83; --av-bad:#ff9da7; --av-pending:#34445e;
 --av-gradient:linear-gradient(110deg,#63e6d4,#73d7ff);
 --body-background-fill:#080f1e; --body-text-color:#edf4fc; --body-text-color-subdued:#a5b5cc;
 --background-fill-primary:#101b2d; --background-fill-secondary:#132139;
 --block-background-fill:#101b2d; --block-border-color:#2a3b53;
 --block-label-text-color:#c2d0e4; --block-title-text-color:#edf4fc;
 --input-background-fill:#0b1425; --input-border-color:#3a4d68;
 --input-text-color:#edf4fc; --input-placeholder-color:#a5b5cc;
 --border-color-primary:#2a3b53; --border-color-accent:#63e6d4; --color-accent:#63e6d4;
 --button-primary-background-fill:#63e6d4; --button-primary-background-fill-hover:#9df3e6;
 --button-primary-text-color:#071724; --button-primary-border-color:#63e6d4;
 --button-secondary-background-fill:#18283f; --button-secondary-text-color:#edf4fc;
 --button-secondary-border-color:#3a4d68; --link-text-color:#73d7ff;
 --table-odd-background-fill:#132139; --table-even-background-fill:#101b2d;
}
body {background:#080f1e!important;margin:0;font-family:ui-sans-serif,system-ui,-apple-system,'Segoe UI',sans-serif}
.gradio-container {font-family:Inter,ui-sans-serif,system-ui,-apple-system,'Segoe UI',sans-serif!important;
 max-width:1240px!important;width:100%!important;box-sizing:border-box!important;
 margin:0 auto!important;padding:0 28px 28px!important;color:var(--av-ink)}
.gradio-container code {overflow-wrap:anywhere;color:#bad7ee;font-size:12px}
.gradio-container a {color:#73d7ff}
.gradio-container button {transition:background .15s,border-color .15s}
.gradio-container :is(button,input,a,textarea,summary,[tabindex]):focus-visible {
 outline:3px solid #73d7ff!important;outline-offset:4px}
.gradio-container .prose {color:var(--av-muted)}
.gradio-container .prose :is(h1,h2,h3,h4,strong) {color:var(--av-ink)}
.av-header {display:flex;align-items:center;justify-content:space-between;gap:16px;padding:25px 0 23px;
 border-bottom:1px solid var(--av-line);margin-bottom:8px;flex-wrap:wrap}
.av-brand {display:flex;align-items:center;gap:12px}
.av-logo {width:42px;height:42px;border-radius:13px;background:var(--av-gradient);display:grid;
 place-items:center;color:#071724;font-weight:850;font-size:18px;letter-spacing:-1px;box-shadow:0 0 30px #63e6d418}
.av-title {margin:0;font-size:24px;font-weight:750;color:var(--av-ink);letter-spacing:-.8px}
.av-title span {color:var(--av-teal)}
.av-sub,.av-note {color:var(--av-muted);font-size:13px;line-height:1.65}
.av-sub {margin:3px 0 0}
.av-badge {display:inline-flex;align-items:center;gap:8px;font-size:11px;font-weight:650;padding:8px 12px;
 border-radius:999px;border:1px solid #3c4b61;color:#e5c98f;background:#1f2230;letter-spacing:.02em}
.av-badge::before {content:'';width:6px;height:6px;border-radius:50%;background:currentColor}
.av-hero {padding:30px 0 24px;display:grid;grid-template-columns:1fr auto;gap:24px;align-items:end}
.av-eyebrow {font-size:10px;font-weight:750;text-transform:uppercase;letter-spacing:.17em;color:var(--av-teal);margin:0 0 10px}
.av-hero h2 {font-size:clamp(25px,3.4vw,40px);letter-spacing:-1.4px;line-height:1.18;margin:0 0 12px;color:var(--av-ink)}
.av-hero h2 span {color:var(--av-teal)}
.av-hero p {max-width:650px}
.av-research-mark {border-left:1px solid var(--av-line);padding-left:22px;font-size:11px;line-height:1.9;
 letter-spacing:.07em;color:var(--av-muted);white-space:nowrap}
.av-notice {display:flex;gap:12px;border:1px solid #51472e;border-radius:12px;background:#211f22;
 padding:12px 16px;color:#e6d7b7;font-size:12px;line-height:1.65;margin:2px 0 16px}
.av-notice strong {color:#f4cc83}
.av-card,.av-panel {background:linear-gradient(145deg,#122037,#0e192b);border:1px solid var(--av-line)!important;
 border-radius:18px!important;padding:22px!important;min-width:0}
.av-panel {margin-top:14px}
.av-panel[data-kind=original] {border-color:#356f67!important}
.av-panel[data-kind=substituted] {border-color:#7c6441!important}
.av-panel[data-kind=substituted] .av-section-title {color:var(--av-warn)}
.av-section-title {font-size:18px;font-weight:650;color:var(--av-ink);margin:0 0 8px;letter-spacing:-.3px}
.av-step {color:var(--av-teal);font-size:11px;letter-spacing:.06em;margin-right:10px}
.av-meta {display:grid;grid-template-columns:1fr 1fr;gap:10px;margin:18px 0}
.av-meta div {border:1px solid var(--av-line);border-radius:10px;padding:12px}
.av-meta dt {font-size:10px;text-transform:uppercase;letter-spacing:.08em;color:var(--av-muted)}
.av-meta dd {font-size:13px;margin:6px 0 0;color:var(--av-ink)}
.av-empty-score {border:1px dashed #3a4d68;border-radius:12px;padding:22px;margin:18px 0;text-align:center}
.av-empty-score .av-score {font-size:38px;font-weight:300;letter-spacing:8px;color:#7d91af}
.av-score-track {height:5px;border-radius:8px;background:#26364e;margin:16px 0 10px;position:relative}
.av-score-track b {display:block;height:100%;background:var(--av-gradient);border-radius:inherit}
.av-score-track i {position:absolute;top:-4px;height:13px;width:2px;background:#f4cc83}
.av-footnote {padding:18px 0 0;margin-top:24px;border-top:1px solid var(--av-line);display:flex;
 justify-content:space-between;gap:12px;flex-wrap:wrap;font-size:11px;color:var(--av-muted)}
.av-flow {display:grid;grid-template-columns:repeat(5,minmax(0,1fr));gap:12px;margin:24px 0}
.av-flow article {padding:20px 14px;border:1px solid var(--av-line);border-radius:14px;background:#101d31}
.av-flow b {display:block;color:var(--av-teal);font-size:11px;margin-bottom:20px}
.av-flow h3 {font-size:14px;color:var(--av-ink);margin-bottom:10px}
.av-flow p {font-size:12px;line-height:1.65;color:var(--av-muted)}
.tab-nav {border-bottom:1px solid var(--av-line)!important;gap:12px!important}
.tab-nav button {font-size:13px!important;padding:12px 14px!important;color:#a5b5cc!important}
.tab-nav button.selected,.tab-nav button[aria-selected=true] {color:#63e6d4!important;border-color:#63e6d4!important}
.av-login {text-align:left;padding:12px 0 20px}
.av-login .av-logo {margin-bottom:22px}
.av-login h1 {font-size:30px;line-height:1.2;letter-spacing:-1px;color:#edf4fc;margin-bottom:14px}
.av-login p {color:#a5b5cc;font-size:13px;line-height:1.7;margin:10px 0}
.av-login .av-eyebrow {font-size:10px;letter-spacing:.15em}
@media(max-width:800px) {.av-hero{grid-template-columns:1fr}.av-research-mark{display:none}
 .av-flow{grid-template-columns:1fr}.av-flow b{margin-bottom:8px}}
@media(max-width:640px) {.gradio-container{padding:0 16px 20px!important}.av-header{padding-top:18px}
 .av-panel,.av-card{padding:16px!important}.av-hero{padding-top:22px}.av-meta{grid-template-columns:1fr}
 .tab-nav{gap:0!important}.tab-nav button{padding:10px 8px!important;font-size:12px!important}}
@media(prefers-reduced-motion:reduce) {*,*::before,*::after{animation:none!important;transition:none!important;scroll-behavior:auto!important}}
"""

BRAND_CSS += RESULT_CSS

LOGIN_CSS = """
.wrap.svelte-14xt79u {min-height:100vh;margin:0!important;padding:32px 20px!important;box-sizing:border-box;
 background:radial-gradient(ellipse at 25% 15%,#16324a 0%,#0b1427 40%,#080f1e 75%)!important}
.wrap.svelte-14xt79u > div {min-width:0!important;width:100%!important;max-width:460px!important;
 border:1px solid #30445e!important;border-radius:22px!important;padding:32px!important;
 background:linear-gradient(145deg,#122037,#0e192b)!important;box-shadow:0 24px 80px #0005}
.wrap.svelte-14xt79u h2 {font-size:14px!important;color:#a5b5cc!important}
.wrap.svelte-14xt79u .auth {margin:0!important}
.wrap.svelte-14xt79u button {border-radius:10px!important;min-height:44px}
.wrap.svelte-14xt79u .creds {color:#ff9da7!important}
@media(max-width:540px) {.wrap.svelte-14xt79u{padding:16px!important}
 .wrap.svelte-14xt79u > div{padding:24px!important}}
"""

LOGIN_HTML = """<div class="av-login"><div class="av-logo" aria-hidden="true">AV</div>
<p class="av-eyebrow">NUS-ISS | GC2</p><h1>Welcome to AValidate.</h1>
<p>Audio-Visual Sound Source Verification</p>
<p>See it. Hear it. Verify the connection.</p>
<p>Sign in with your assigned credentials. Uploads are isolated to your login session
and removed after checking or expiry.</p></div>"""


def header_html(model_ready: bool) -> str:
    # Finding artifacts at startup never grants a release approval.
    return """<header class="av-header"><div class="av-brand">
<div class="av-logo" aria-hidden="true">AV</div><div><h1 class="av-title">A<span>Validate</span></h1>
<p class="av-sub">Audio-Visual Sound Source Verification</p></div></div>
<span class="av-sub">NUS-ISS | GC2</span></header>
<section class="av-hero"><div><p class="av-eyebrow">AUDIO + VISION / ONE PERSPECTIVE</p>
<h2>See it. Hear it.<br><span>Verify the connection.</span></h2>
<p class="av-sub">Upload a ten-second video or record locally to see event recognition,
temporal AV presence and audio-visual correspondence predictions.</p></div></section>"""


READINESS_HTML = """<aside class="av-notice" role="status"><span aria-hidden="true">◈</span>
<div><strong>MODEL_NOT_READY · Predictions disabled</strong><br>Media admission is available.
Model verification awaits independent preprocessing, numerical and release approval.</div></aside>"""

HOW_HTML = """<section class="av-panel"><p class="av-eyebrow">THE PIPELINE</p>
<h2 class="av-section-title">Two streams. A pairing decision.</h2><p class="av-note">
The trained architecture compares representations across ten corresponding time segments.</p>
<div class="av-flow"><article><b>01 →</b><h3>Audio + video</h3><p>A validated ten-second clip with one audio and one video track.</p></article>
<article><b>02 →</b><h3>Encode</h3><p>BEATs iter3 + trained LoRA: 10 × 256. Trained ResNet-18: 10 × 512.</p></article>
<article><b>03 →</b><h3>Align</h3><p>Match corresponding time segments. Timing diagnostics alone do not approve a decoder.</p></article>
<article><b>04 →</b><h3>Fuse</h3><p>Frozen E3 concatenation + MLP combines representations. A2 attention is a comparison.</p></article>
<article><b>05</b><h3>Decide</h3><p>Apply the frozen operating point: original-like or audio-substituted-like pairing.</p></article></div>
<p class="av-note">Pairing correspondence does not prove that a visible object physically generated a sound.
The sigmoid score is uncalibrated. The optional trained Head 2 predicts annotated AVE event presence
per second; its separate threshold and evaluated scope do not change E3 or public readiness.</p></section>"""

ABOUT_MD = """### Model information

| Component | Frozen architecture | Output |
| --- | --- | --- |
| Audio | BEATs iter3 + trained LoRA and classification head | 10 × 256 embeddings; 10 × 29 event probabilities |
| Video | Trained ResNet-18 | 10 × 512 embeddings; 10 × 29 event probabilities |
| Primary fusion | E3 concatenation + MLP | One uncalibrated correspondence score |
| Optional temporal presence | Separately trained MLP + bidirectional GRU | Ten uncalibrated binary scores for annotated joint AVE event presence |
| Comparison | A2 attention | Separate pairing score |

**Public status: MODEL_NOT_READY.** Historical audio/video exact parity remains unresolved.
Numerical tolerance acceptance is inactive; general decoder/shared-clock approval and an
independently reviewed release decision are pending. Adding an adapter does not grant approval.

In the private authenticated product, Head 1 shows each encoder's original per-second
event softmax evidence and mean event rankings. These scores are uncalibrated. Head 2
uses its separately versioned trained artifact when privately provisioned; otherwise it is unavailable.
Its evaluation covers original matched AVE clips, not swapped/live temporal ground truth.
Head 3 retains the frozen E3 pairing decision. Ordinary public upload/live gates stay closed.
TEST data must not be used for tuning.
"""

FOOTER_HTML = """<footer class="av-footnote"><span>AValidate · NUS-ISS | GC2</span>
<span>Session-isolated uploads · Local preview · No analytics</span></footer>"""
