"""Explicitly isolated general-media research panel. No public result promotion."""
from html import escape

from deployment.general_inference import ResearchPrediction, response_for_research, error


def research_card(result):
    if isinstance(result,dict) and result.get('status')=='REJECTED':
        try:r=response_for_research(result)
        except Exception:r=error('GENERAL_INFERENCE_FAILED')
        code=r['error_code']
        messages={'INVALID_DURATION':'Use a complete 10-second video.',
            'GENERAL_VIDEO_INCOMPLETE':'Ten complete video seconds are required.',
            'GENERAL_AUDIO_INCOMPLETE':'Ten complete audio seconds are required.',
            'UPLOAD_VALIDATION_FAILED':'The file could not be read as supported MP4/MOV media.',
            'UNSUPPORTED_EXTENSION':'Choose an MP4 or MOV file with H.264 video and AAC audio.',
            'PRODUCT_WORKER_NOT_READY':'The isolated model worker is unavailable. Ask the operator to start it.',
            'UPLOAD_CANCELLED':'Verification was cancelled. Upload again to start a new request.',
            'RESOURCE_BUSY':'Another verification is processing. Please try again shortly.'}
        return ('<section class="av-result" role="alert"><h4>Verification unavailable</h4><p>'+
            escape(messages.get(code,'The input was rejected; no prediction was produced.'))+
            '</p><p class="av-note">No score is shown. '+escape(code)+'</p></section>')
    try:
        r = response_for_research(result)
        p = ResearchPrediction(**{k:r[k] for k in ResearchPrediction.__dataclass_fields__})
    except Exception:
        return '<section class="av-result" role="status"><h4>Research inference unavailable</h4><p>No score is shown. Inspect the sanitized response.</p></section>'
    return (f'<section class="av-panel" role="status" aria-live="polite"><p class="av-eyebrow">RESEARCH ONLY / NOT APPROVED</p>'
        f'<h3 class="av-section-title">{escape(r["classification"])}</h3>'
        f'<p class="av-note">Actual raw E3 score <strong>{p.score}</strong> · frozen threshold {p.threshold}<br>'
        'At or above the threshold: Original-like. Below: Audio-substituted-like.<br>'
        f'E3 / frozen concatenation + MLP · {p.processing_seconds:.3f} s<br>Model <code>{p.model_sha256}</code><br>'
        f'Source <code>{p.source_sha256}</code><br>Profile <code>{p.profile_sha256}</code></p>'
        '<p class="av-note">Uncalibrated correspondence score. This does not prove physical sound causation. '
        'Historical parity, numerical acceptance and independent release authorization remain unresolved. '
        'Ordinary public prediction gates remain closed.</p></section>')


def build_general_research_section(gr, service, *, private_demo=False):
    if private_demo:
        gr.Markdown('### Verify a video\nUpload a complete **10-second MP4 or MOV** with H.264 video and AAC audio, '
            'up to 50 MiB. Select **Verify** to run the original BEATs + LoRA, ResNet-18 and frozen E3 models. '
            'Unsupported formats or incomplete/ambiguous clocks are rejected. '
            '**Private demonstration; not independently approved for public inference.**')
    else:gr.Markdown('### General media research — NOT APPROVED\n'
        '**Isolated research/test path.** Admitted new media may run the authentic models here, without TRAIN identities. '
        'CFR, zero stream origins, mono/stereo, 8-bit YUV420, no rotation/HDR, at most 1920×1080 and 120 fps. '
        'Existing format/codec/duration/50 MiB admission rules also apply. Files lacking explicit stream durations are rejected. '
        'Scores here do not authorize public inference or establish historical numerical compatibility.')
    file = gr.File(label='Upload a 10-second MP4 or MOV',file_types=['.mp4','.mov'],
                   file_count='single',type='filepath',elem_id='av-general-research-source')
    handle = gr.Textbox(visible=False)
    file.change(fn=None,inputs=file,outputs=handle,queue=False,
                js="(f) => f && typeof f.path === 'string' ? f.path : ''")
    button = gr.Button('Verify',variant='primary',interactive=False)
    cancel = gr.Button('Cancel / reset result')
    card = gr.HTML('<p class="av-note">No general-media research inference has run.</p>',elem_id='av-general-result')
    raw = gr.JSON(label='Research response; production predictions disabled',elem_id='av-general-response')

    def infer_general(owned_handle, request: gr.Request):
        yield '<p class="av-note" role="status">Admitting, decoding and processing research media...</p>',None,gr.update(interactive=False)
        try:
            r = response_for_research(service.submit_owned(owned_handle,request))
        except Exception:
            r = error('GENERAL_INFERENCE_FAILED')
        yield research_card(r),r,gr.update(interactive=False)

    event = button.click(infer_general,inputs=handle,outputs=[card,raw,button],api_name=None,
                         api_visibility='private',concurrency_limit=1,trigger_mode='once',show_progress='full')
    def reset_selection(owned_handle, request: gr.Request):
        if hasattr(service,'cancel_owned'):service.cancel_owned(request)
        return '<p class="av-note">Upload a file and select Verify for a fresh result.</p>',None,gr.update(interactive=bool(owned_handle))

    handle.change(fn=reset_selection,inputs=handle,
        outputs=[card,raw,button],api_name=None,api_visibility='private',queue=False,cancels=[event])
    cancel.click(reset_selection,inputs=handle,outputs=[card,raw,button],api_name=None,api_visibility='private',queue=False,cancels=[event])
