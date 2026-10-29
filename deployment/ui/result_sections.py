"""Three objective sections; encoder evidence is separate from fused presence."""
from html import escape
from html.parser import HTMLParser
import json

import numpy as np

from deployment.event_evidence import TEMPORAL_UNAVAILABLE, summary

RESULT_CSS = """
.av-heads {display:grid;gap:14px;white-space:normal;text-align:left}
.av-research-result {grid-column:1/-1;min-width:0}
.av-head {min-width:0;border:1px solid var(--av-line,#2a3b53);border-radius:16px;
 padding:20px;background:var(--av-card,#101b2d);overflow-wrap:anywhere}
.av-head h4 {margin:4px 0 12px;font-size:19px;color:var(--av-ink,#edf4fc)}
.av-head h5 {margin:0 0 12px;font-size:14px;color:var(--av-ink,#edf4fc)}
.av-head p {font-size:13px;line-height:1.65;color:var(--av-muted,#a5b5cc);margin:8px 0}
.av-head-label {font-size:10px;letter-spacing:.13em;font-weight:700;color:var(--av-teal,#63e6d4)}
.av-event-columns {display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:20px}
.av-event-list {list-style:none;margin:0;padding:0;display:grid;gap:8px}
.av-event-list li {display:grid;grid-template-columns:minmax(0,1fr) auto;gap:6px 12px;font-size:13px}
.av-event-list strong {font-variant-numeric:tabular-nums;font-weight:500;color:var(--av-ink,#edf4fc)}
.av-event-list meter {grid-column:1/-1;width:100%;height:6px;accent-color:var(--av-teal,#63e6d4)}
.av-event-list meter::-webkit-meter-bar {background:#23334a;border:0}
.av-event-list meter::-webkit-meter-optimum-value {background:#63e6d4}
.av-unavailable {border-style:dashed;background:#121d2c}
.av-unavailable strong {color:var(--av-warn,#f4cc83);font-size:14px}
.av-presence-grid {display:grid;grid-template-columns:repeat(5,minmax(0,1fr));gap:9px;margin:14px 0}
.av-presence-cell {border:1px solid #3c5069;border-radius:10px;padding:10px 5px;text-align:center;font-size:11px;background:#162238}
.av-presence-cell[data-state=present] {border-color:#448779;background:#15352f}
.av-presence-cell b,.av-presence-cell span {display:block;margin:5px 0;font-variant-numeric:tabular-nums}
.av-presence-cell b {color:var(--av-ink,#edf4fc)}
.av-correspondence {border-color:#3a817f;background:linear-gradient(120deg,#102d32,#101b2d)}
.av-correspondence h4 {font-size:26px;letter-spacing:-.5px}
.av-raw-score {font-size:22px;font-variant-numeric:tabular-nums;color:var(--av-ink,#edf4fc)}
.av-head details {margin-top:14px;border-top:1px solid var(--av-line,#2a3b53);padding-top:12px}
.av-head summary {font-size:13px;font-weight:600;cursor:pointer;color:var(--av-green,#73d7ff)}
.av-table-scroll {max-width:100%;overflow-x:auto;margin-top:12px}
.av-event-timeline {width:100%;border-collapse:collapse;font-size:12px;min-width:560px}
.av-event-timeline th,.av-event-timeline td {padding:8px;text-align:left;border-bottom:1px solid #2a3b53}
.av-event-timeline th {color:var(--av-muted,#a5b5cc);font-weight:500}
.av-event-timeline td {color:var(--av-ink,#edf4fc);font-variant-numeric:tabular-nums}
.av-heads pre {white-space:pre-wrap;font-size:11px;max-height:420px;overflow:auto;color:var(--av-muted,#a5b5cc)}
@media(max-width:640px) {.av-event-columns{grid-template-columns:1fr}.av-head{padding:16px}
 .av-correspondence h4{font-size:22px}.av-raw-score{font-size:18px}.av-presence-grid{grid-template-columns:repeat(2,minmax(0,1fr))}}
"""


def event_score(value):
    # Preserve visible nonzero evidence below the fixed-decimal display scale.
    return f'{value:.3e}' if 0 < value < 1e-6 else f'{value:.6f}'


def temporal_section(prediction=None):
    if prediction is not None:
        cells=[]
        for second,(score,present) in enumerate(zip(prediction['scores'],prediction['presence'])):
            label='Joint presence detected' if present else 'Background / not detected'
            cells.append(f'<div class="av-presence-cell" data-state="{"present" if present else "background"}" '
                f'aria-label="Second {second}: {label}, score {score}"><span>{second}–{second+1} s</span>'
                f'<b>{label}</b><span>{event_score(score)}</span></div>')
        detail = '<details><summary>Per-second values</summary><pre>' + escape(json.dumps(prediction['scores'], indent=2)) + '</pre></details>'
        return ('<section class="av-head" data-head="temporal"><span class="av-head-label">HEAD 2 · TEMPORAL AV PRESENCE</span>'
                '<h4>Temporal AV presence</h4><p>Genuine per-second predictions for the presence '
                'of annotated joint AVE events. Background denotes the model’s background class.</p>'
                f'<div class="av-presence-grid">{"".join(cells)}</div>'
                f'<p>{escape(prediction["model_version"])} · Head 2 threshold: {prediction["threshold"]}. '
                'Scores are uncalibrated; they are not calibrated presence probabilities.</p>'
                '<p>Trained and evaluated on original matched AVE clips. Audio-swapped pairs and browser-live '
                'inputs have no new temporal ground truth and remain outside this evaluated scope. '
                'This does not prove physical sound causation or absence of every unannotated event.</p>'+detail+'</section>')
    return ('<section class="av-head av-unavailable" data-head="temporal"><span class="av-head-label">HEAD 2</span>'
            '<h4>Per-second audible-and-visible presence</h4>'
            f'<strong>{escape(TEMPORAL_UNAVAILABLE)}</strong><p>No verified trained temporal artifact is loaded. '
            'Encoder event evidence and E3 correspondence do not supply this classification.</p></section>')


def empty_results(message):
    return ('<div class="av-heads"><section class="av-head" data-head="events"><span class="av-head-label">HEAD 1</span>'
            '<h4>Event recognition / evidence</h4><p>Audio and visual evidence awaits a valid inference.</p></section>'
            + temporal_section() + '<section class="av-head av-correspondence" data-head="correspondence">'
            '<span class="av-head-label">HEAD 3</span><h4>Audio-visual correspondence</h4>'
            f'<p>{escape(message)} No score is shown.</p></section></div>')


def event_section(evidence):
    intro = ('<section class="av-head" data-head="events"><span class="av-head-label">HEAD 1</span>'
             '<h4>Event recognition</h4><p>Model predictions, not ground-truth annotations.</p>')
    if evidence is None:
        return intro+'<p>Encoder event evidence is not available for this result.</p></section>'
    columns=[]
    for modality,title in (('audio','Top audio events · BEATs + LoRA'),('video','Top visual evidence · ResNet-18')):
        rows,background=summary(evidence,modality)
        items=''.join(f'<li><span>{escape(label)}</span><strong>{event_score(score)}</strong>'
                      f'<meter min="0" max="1" value="{score}" aria-label="{escape(label,quote=True)} mean softmax score"></meter></li>'
                      for label,score in rows)
        columns.append(f'<div><h5>{title}</h5><ul class="av-event-list">{items}</ul>'
                       f'<p>Mean background score: {event_score(background)}</p></div>')
    rows=[]
    for second in range(10):
        cells=[]
        for modality in ('audio','video'):
            scores=evidence[modality]['scores'][second]
            index=int(np.argmax(scores))
            cells.append(f'<td>{escape(evidence["class_names"][index])}</td><td>{event_score(scores[index])}</td>')
        rows.append(f'<tr><td>{second}–{second+1} s</td>{"".join(cells)}</tr>')
    return (intro+'<p title="Means of the ten original softmax rows; no background exclusion or renormalization in the scores.">'
            'Top 5 event classes by mean softmax score across the window. Scores are uncalibrated; '
            'rankings are evidence, not thresholded detections. Background is reported separately.</p>'
            f'<div class="av-event-columns">{"".join(columns)}</div>'
            '<details class="av-event-details"><summary>10-second event-evidence timeline</summary>'
            '<p>Window-relative encoder bins. Audio is pooled per second; video uses one sampled frame per bin. '
            'Top class per modality includes background. This is not a fused temporal-presence prediction. '
            'Actual frame timestamps are retained in advanced diagnostics.</p><div class="av-table-scroll">'
            '<table class="av-event-timeline"><thead><tr><th scope="col">Window bin</th><th scope="col">Audio evidence</th>'
            '<th scope="col">Score</th><th scope="col">Visual evidence</th><th scope="col">Score</th></tr></thead>'
            f'<tbody>{"".join(rows)}</tbody></table></div></details></section>')


def three_head_result(result):
    """Called only after the result ABI and event evidence are validated."""
    return ThreeHeadResults.render(result)


class _ViewParser(HTMLParser):
    """Serialize trusted, escaped component markup as inert DOM instructions."""
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.root = {'tag': 'div', 'attrs': {}, 'children': []}
        self.stack = [self.root]

    def handle_starttag(self, tag, attrs):
        node = {'tag': tag, 'attrs': dict(attrs), 'children': []}
        self.stack[-1]['children'].append(node)
        if tag not in ('br',):
            self.stack.append(node)

    def handle_endtag(self, tag):
        if len(self.stack) > 1 and self.stack[-1]['tag'] == tag:
            self.stack.pop()

    def handle_data(self, data):
        self.stack[-1]['children'].append(data)


class ThreeHeadResults:
    """Single presentation component for authenticated upload and live results."""
    @staticmethod
    def view(result):
        parser = _ViewParser()
        parser.feed(ThreeHeadResults.render(result))
        return parser.root

    @staticmethod
    def render(result):
        decision = 'MATCH' if result['score'] >= result['threshold'] else 'MISMATCH'
        raw = escape(json.dumps(result, indent=2))
        return _render_result(result, decision, raw)


def _render_result(result, decision, raw):
    return ('<div class="av-heads" role="status" aria-live="polite">'+event_section(result['event_predictions'])
            +temporal_section(result['temporal_predictions'])+'<section class="av-head av-correspondence" data-head="correspondence">'
            '<span class="av-head-label">HEAD 3 · AUDIO-VISUAL CORRESPONDENCE</span>'
            f'<h4>{decision}</h4><p>{escape(result["classification"])}</p><p>Actual raw E3 score</p>'
            f'<strong class="av-raw-score">{result["score"]}</strong>'
            f'<p title="The original trained E3 operating point is unchanged.">Frozen threshold: {result["threshold"]}<br>'
            'At or above threshold: Original-like. Below: Audio-substituted-like.</p>'
            f'<p>{escape(result["model_version"])} · {result["processing_seconds"]:.3f} s</p>'
            '<p>Uncalibrated correspondence score. It is not a calibrated confidence percentage or proof of physical '
            'sound causation.</p><details><summary>Advanced diagnostics · raw scores and model provenance</summary>'
            f'<pre data-av-raw-result="">{raw}</pre></details></section></div>')
