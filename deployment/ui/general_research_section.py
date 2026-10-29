"""Explicitly isolated general-media research panel. No public result promotion."""
from html import escape

from deployment.general_inference import ResearchPrediction, response_for_research, error


def research_card(result):
    try:
        r = response_for_research(result)
        p = ResearchPrediction(**{k:r[k] for k in ResearchPrediction.__dataclass_fields__})
    except Exception:
        return '<section class="av-result" role="status"><h4>Research inference unavailable</h4><p>No score is shown. Inspect the sanitized response.</p></section>'
    return (f'<section class="av-panel" role="status" aria-live="polite"><p class="av-eyebrow">RESEARCH ONLY / NOT APPROVED</p>'
        f'<h3 class="av-section-title">{escape(r["classification"])}</h3>'
        f'<p class="av-note">Actual raw E3 score <strong>{p.score}</strong> · frozen threshold {p.threshold}<br>'
        f'E3 / frozen concatenation + MLP · {p.processing_seconds:.3f} s<br>Model <code>{p.model_sha256}</code><br>'
        f'Source <code>{p.source_sha256}</code><br>Profile <code>{p.profile_sha256}</code></p>'
        '<p class="av-note">Uncalibrated correspondence score. This does not prove physical sound causation. '
        'Historical parity, numerical acceptance and independent release authorization remain unresolved. '
        'Public and live model predictions remain disabled.</p></section>')


def build_general_research_section(gr, service):
    gr.Markdown('### General media research — NOT APPROVED\n'
        '**Isolated research/test path.** Admitted new media may run the authentic models here, without TRAIN identities. '
        'CFR, zero stream origins, mono/stereo, 8-bit YUV420, no rotation/HDR, at most 1920×1080 and 120 fps. '
        'Existing format/codec/duration/50 MiB admission rules also apply. Files lacking explicit stream durations are rejected. '
        'Scores here do not authorize public inference or establish historical numerical compatibility.')
    file = gr.File(label='Research audiovisual file',file_types=['.mp4','.mov','.mkv','.webm'],
                   file_count='single',type='filepath',elem_id='av-general-research-source')
    handle = gr.Textbox(visible=False)
    file.change(fn=None,inputs=file,outputs=handle,queue=False,
                js="(f) => f && typeof f.path === 'string' ? f.path : ''")
    button = gr.Button('Run research E3 · NOT APPROVED',variant='primary')
    card = gr.HTML('<p class="av-note">No general-media research inference has run.</p>')
    raw = gr.JSON(label='Research response; production predictions disabled')

    def infer_general(owned_handle, request: gr.Request):
        yield '<p class="av-note" role="status">Admitting, decoding and processing research media...</p>',None,gr.update(interactive=False)
        try:
            r = response_for_research(service.submit_owned(owned_handle,request))
        except Exception:
            r = error('GENERAL_INFERENCE_FAILED')
        yield research_card(r),r,gr.update(interactive=False)

    event = button.click(infer_general,inputs=handle,outputs=[card,raw,button],api_name=None,
                         api_visibility='private',concurrency_limit=1,trigger_mode='once',show_progress='full')
    handle.change(fn=lambda: ('<p class="av-note">Selection changed. Run again for a fresh result.</p>',None,gr.update(interactive=True)),
        outputs=[card,raw,button],api_name=None,api_visibility='private',queue=False,cancels=[event])
