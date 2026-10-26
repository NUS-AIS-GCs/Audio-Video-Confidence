"""Upload Verification section (owner: Swaraj, upload interface / preview / results dashboard).

This keeps the existing Phase 8 upload admission controls unchanged inside the new layout, so the
secure callback (`DeploymentUI.submit`) and its private endpoint stay exactly as they were. Replace the
body of `build_upload_section` with the full upload UI; keep the signature so the layout does not change.
"""


def build_upload_section(gr, ui) -> None:
    gr.Markdown("### Upload verification\nCheck one recorded 10-second clip with sound.")
    uploaded = gr.File(
        label="One 10-second audio-video file (MP4, MOV, MKV, or WebM)",
        file_count="single",
        file_types=[".mp4", ".mov", ".mkv", ".webm"],
        type="filepath",
    )
    submit = gr.Button("Validate upload (inference disabled)", variant="secondary")
    output = gr.JSON(label="Admission status; predictions are disabled")
    submit.click(
        fn=ui.submit,
        inputs=uploaded,
        outputs=output,
        api_name=None,
        api_visibility="private",
        concurrency_limit=1,
    )
