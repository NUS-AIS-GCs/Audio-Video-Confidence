"""AValidate branding: one stylesheet and the page header, shared by every section.

Colours follow the team deck (teal to green). Gradio's own CSS variables are overridden so built-in
components (buttons, tabs, inputs) pick up the brand in light and dark mode.
"""

BRAND_CSS = """
:root, .gradio-container {
  --av-teal: #0e9f9a; --av-green: #2fbf71; --av-ink: #10222b; --av-muted: #5b6b73;
  --av-card: #ffffff; --av-line: #dbe5e8; --av-bg: #f3f7f8;
  --av-ok: #2fbf71; --av-warn: #e3a008; --av-bad: #e0484f; --av-pending: #c9d4d8;
  --av-gradient: linear-gradient(90deg, var(--av-teal), var(--av-green));
  --color-accent: var(--av-teal);
  --button-primary-background-fill: var(--av-gradient);
  --button-primary-background-fill-hover: linear-gradient(90deg, #0b8a86, #27a862);
  --button-primary-text-color: #ffffff;
  --body-background-fill: var(--av-bg);
}
.dark, .dark .gradio-container {
  --av-ink: #e6f1f3; --av-muted: #9db0b7; --av-card: #14232a; --av-line: #24363e; --av-bg: #0b161b;
  --av-pending: #2e434b;
  --body-background-fill: var(--av-bg);
}
.gradio-container { max-width: 1180px !important; width: 100% !important; box-sizing: border-box !important;
  margin: 0 auto !important; padding: 0 16px !important; }
.gradio-container code { overflow-wrap: anywhere; }
.av-header { display: flex; align-items: center; justify-content: space-between; gap: 16px;
  padding: 22px 0 14px; border-bottom: 1px solid var(--av-line); margin-bottom: 8px; flex-wrap: wrap; }
.av-brand { display: flex; align-items: center; gap: 12px; }
.av-logo { width: 40px; height: 40px; border-radius: 12px; background: var(--av-gradient);
  display: grid; place-items: center; color: #fff; font-weight: 800; font-size: 20px; letter-spacing: -1px; }
.av-title { margin: 0; font-size: 26px; font-weight: 800; color: var(--av-ink); letter-spacing: -0.5px; }
.av-title span { background: var(--av-gradient); -webkit-background-clip: text; background-clip: text; color: transparent; -webkit-text-fill-color: transparent; }
.av-sub { margin: 2px 0 0; color: var(--av-muted); font-size: 14px; }
.av-badge { font-size: 12px; font-weight: 600; padding: 6px 12px; border-radius: 999px;
  border: 1px solid var(--av-line); color: var(--av-muted); background: var(--av-card); white-space: nowrap; }
.av-badge[data-kind="off"]::before { content: ""; display: inline-block; width: 8px; height: 8px; border-radius: 50%;
  background: var(--av-warn); margin-right: 8px; vertical-align: middle; }
.av-card { background: var(--av-card); border: 1px solid var(--av-line); border-radius: 16px; padding: 18px; }
.av-note { color: var(--av-muted); font-size: 13px; line-height: 1.5; }
.av-section-title { font-size: 18px; font-weight: 700; color: var(--av-ink); margin: 6px 0 4px; }
@media (max-width: 640px) {
  .av-title { font-size: 21px; } .av-header { padding-top: 14px; } .av-card { padding: 14px; }
}
"""


def header_html(model_ready: bool) -> str:
    badge = ("Model ready" if model_ready else "Model not ready · predictions disabled")
    return f"""
<header class="av-header">
  <div class="av-brand">
    <div class="av-logo" aria-hidden="true">AV</div>
    <div>
      <h1 class="av-title">A<span>Validate</span></h1>
      <p class="av-sub">Audio-visual sound-source verification</p>
    </div>
  </div>
  <span class="av-badge" data-kind="{'on' if model_ready else 'off'}">{badge}</span>
</header>"""


ABOUT_MD = """
### How AValidate works
AValidate checks whether the sound in a 10-second clip comes from what is on screen. It splits the clip
into ten one-second segments, encodes the audio (BEATs + LoRA) and the frames (ResNet-18), and a fusion
model compares the two streams.

**Upload verification** checks one recorded 10-second clip. **Live verification** records from your
webcam and microphone and keeps the latest ten seconds, aligned second by second.

Predictions stay switched off until the trained encoders are connected and verified against the
training exports. Until then the app shows checks and diagnostics only.
"""
