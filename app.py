"""Streamlit Web Application for AValidate: Intelligent Audio-Visual Source Matching."""
import streamlit as st
import tempfile
import os
from pathlib import Path
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from src.pipeline import AValidatePipeline
from src.config import AVE_CLASSES, TAU_UPPER, TAU_LOWER, VIDEOS_DIR
from src.utils.data_loader import get_ave_index

st.set_page_config(
    page_title="AValidate | Audio-Visual Source Verification",
    page_icon="🎬",
    layout="wide"
)

# Custom Styling to match Slide 10 layout and dark-teal accents
st.markdown("""
<style>
    .metric-card {
        background-color: #0f2537;
        border-radius: 10px;
        padding: 20px;
        color: white;
    }
    .track-label {
        font-weight: 600;
        font-size: 15px;
    }
    .evidence-box {
        display: inline-block;
        padding: 4px 8px;
        margin: 2px;
        border-radius: 4px;
        font-family: monospace;
    }
    .demo-pill {
        display: inline-block;
        background-color: #d97706;
        color: #ffffff;
        padding: 2px 8px;
        border-radius: 12px;
        font-size: 11px;
        font-weight: 700;
        letter-spacing: 0.5px;
        text-transform: uppercase;
        margin-left: 8px;
        vertical-align: middle;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_pipeline():
    return AValidatePipeline()

pipeline = load_pipeline()
ave_index = get_ave_index()

# Header
st.title("🎬 AValidate: Intelligent Audio-Visual Source Matching")
st.markdown("*Does what you hear make sense given what you see?*")

# PROMINENT DEMO DISCLAIMER BANNER
st.warning(
    "⚠️ **PROTOTYPE DEMO MODE — PLACEHOLDER OUTPUTS**\n\n"
    "Trained neural network models are currently being trained and integrated. All predictions, confidence numbers, "
    "temporal presence grids, and correspondence scores displayed below are **simulated / placeholder outputs** "
    "for interactive UI demonstration only. **Please do not mistake these numbers for experimental results.**"
)

# Top Selection: Upload or Sample Video
col_mode1, col_mode2 = st.columns([1, 1])

with col_mode1:
    source_mode = st.radio(
        "Choose Input Source:",
        ["Select Sample from AVE Dataset (Instant Test)", "Upload Custom Video"],
        horizontal=True
    )

selected_video_path = None
uploaded_temp_file = None

if source_mode == "Select Sample from AVE Dataset (Instant Test)":
    sample_videos = ave_index.get_sample_videos_per_class()
    if sample_videos:
        chosen_cat = st.selectbox("Select Event Category to Test:", sorted(list(sample_videos.keys())))
        vid_id = sample_videos[chosen_cat]
        selected_video_path = str(VIDEOS_DIR / f"{vid_id}.mp4")
        st.caption(f"Loaded AVE Benchmark Clip: `{vid_id}.mp4` ({chosen_cat})")
    else:
        st.warning("No sample videos found in AVE_Dataset/AVE directory.")
else:
    uploaded_file = st.file_uploader("Upload a 10-second video (.mp4, .avi, .mkv)", type=["mp4", "avi", "mkv"])
    if uploaded_file is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
            tmp.write(uploaded_file.read())
            selected_video_path = tmp.name
            uploaded_temp_file = tmp.name

# Main Analysis Section
if selected_video_path and os.path.exists(selected_video_path):
    # Interactive Controls
    ctrl_col, _ = st.columns([4, 6])
    with ctrl_col:
        swap_audio = st.toggle("🔀 Simulate Swapped Audio (Mismatch Counterfactual Test)", value=False)
        if swap_audio:
            st.caption("⚠️ **Counterfactual Mode Active:** Audio track is swapped with an unrelated sound to test mismatch rejection.")

    with st.spinner("Analyzing audio-visual synchrony and temporal evidence..."):
        result = pipeline.predict(selected_video_path, swap_audio_test=swap_audio)

    # Upper Deck: Video Player (Left) and Decision Card (Right)
    col_video, col_decision = st.columns([5, 5])

    with col_video:
        st.markdown("### INPUT VIDEO")
        st.video(selected_video_path)
        st.caption("Synchronized 10-second audio-video input.")
        if not swap_audio:
            st.success("🟢 **Video Branch Active:** Frame extraction, CLIP embeddings, and visual presence mapped.")
        else:
            st.error("🔴 **Audio Swapped:** Testing system behavior under cross-modal mismatch.")

    with col_decision:
        st.markdown("### DECISION <span class='demo-pill'>DEMO MOCK</span>", unsafe_allow_html=True)
        st.caption("⚠️ *Results below are simulated heuristics pending trained model integration.*")
        
        # Decision Badge
        if result.correspondence_band == "Strong Match":
            st.success("## 🟢 STRONG MATCH")
        elif result.correspondence_band == "Weak / Uncertain":
            st.warning("## 🟡 WEAK / UNCERTAIN")
        else:
            st.error("## 🔴 MISMATCH")

        # Large Metric Typography
        st.markdown(f"## **{result.correspondence_score:.2f}** <span style='font-size:18px; color:gray;'>correspondence score</span>", unsafe_allow_html=True)
        st.markdown(f"**Predicted Event:** `{result.predicted_event}` (Confidence: {result.event_confidence * 100:.1f}%)")

        # Confidence Bands Legend
        with st.expander("Confidence Bands Legend (Validation Calibrated)", expanded=True):
            st.markdown(f"""
            - 🟢 **Strong match:** score > {TAU_UPPER:.2f}
            - 🟡 **Weak / Uncertain:** {TAU_LOWER:.2f} ≤ score ≤ {TAU_UPPER:.2f}
            - 🔴 **Mismatch:** score < {TAU_LOWER:.2f}  
            *Thresholds are empirically calibrated on the AVE validation split.*
            """)

        # Top-5 Predicted Distribution Drawer
        with st.expander("Top-5 Predicted Classes (Head 1)"):
            df_top5 = pd.DataFrame(result.top5_classes)
            fig_bar = go.Figure(go.Bar(
                x=df_top5["prob"],
                y=df_top5["class"],
                orientation='h',
                marker=dict(color='#0d9488')
            ))
            fig_bar.update_layout(
                margin=dict(l=0, r=0, t=10, b=10),
                height=180,
                xaxis=dict(range=[0, 1], title="Probability"),
                yaxis=dict(autorange="reversed")
            )
            st.plotly_chart(fig_bar, use_container_width=True)

    st.divider()

    # Lower Deck: Per-Second Evidence (10-Second Temporal Localization)
    st.markdown("### PER-SECOND EVIDENCE (10-Second Temporal Localization) <span class='demo-pill'>DEMO TRACKS</span>", unsafe_allow_html=True)
    st.caption("⚠️ *Tracks below display prototype visualization layout with mock temporal evidence.*")

    # 1. Ten Sampled Frames Filmstrip
    cols_thumb = st.columns(10)
    for i, col in enumerate(cols_thumb):
        with col:
            st.image(result.sampled_frames[i], use_container_width=True)
            st.caption(f"**{i}s**")

    # 2. Three Evidence Tracks (Slide 10 Layout)
    st.markdown("---")

    def format_blocks(arr, active_symbol, inactive_symbol="⬜"):
        return " &nbsp; ".join([active_symbol if val == 1 else inactive_symbol for val in arr])

    col_track_labels, col_track_grids = st.columns([2, 8])

    with col_track_labels:
        st.write(f"**Visual:** `{result.predicted_event}`")
        st.write("**Audio Track:**")
        st.write("**Audible & visible:**")

    with col_track_grids:
        # Visual Track
        v_blocks = format_blocks(result.visual_presence, "🟩")
        st.markdown(f"{v_blocks}", unsafe_allow_html=True)
        
        # Audio Track
        a_blocks = format_blocks(result.audio_presence, "🟦")
        st.markdown(f"{a_blocks}", unsafe_allow_html=True)
        
        # Joint Track (Audible & Visible)
        j_blocks = format_blocks(result.audible_and_visible, "⭐")
        st.markdown(f"{j_blocks}", unsafe_allow_html=True)

    # Explainability & Diagnostic Drawer
    with st.expander("🔍 Deep Explainability & Trajectory Analysis"):
        st.caption("ℹ️ *Trajectories and attribution percentages are currently prototype mock data.*")
        st.markdown("#### 1. Second-by-Second Cosine Similarity Trajectory")
        fig_traj = go.Figure()
        fig_traj.add_trace(go.Scatter(
            x=[f"{i}s" for i in range(10)],
            y=result.similarity_trajectory,
            mode='lines+markers',
            name='Audio-Visual Similarity',
            line=dict(color='#06b6d4', width=3)
        ))
        fig_traj.add_hline(y=TAU_UPPER, line_dash="dash", line_color="green", annotation_text="Strong Match Threshold")
        fig_traj.add_hline(y=TAU_LOWER, line_dash="dash", line_color="red", annotation_text="Mismatch Threshold")
        fig_traj.update_layout(height=240, margin=dict(l=0, r=0, t=20, b=20), yaxis=dict(range=[0, 1]))
        st.plotly_chart(fig_traj, use_container_width=True)

        st.markdown("#### 2. Modality Attribution Breakdown")
        col_m1, col_m2, col_m3 = st.columns(3)
        col_m1.metric("Visual Branch Confidence", f"{result.modality_attribution['visual'] * 100:.1f}%")
        col_m2.metric("Audio Branch Confidence", f"{result.modality_attribution['audio'] * 100:.1f}%")
        col_m3.metric("Fused Joint Confidence", f"{result.modality_attribution['fused'] * 100:.1f}%")

        if swap_audio:
            st.error("⚠️ **Counterfactual Explanation:** Cross-modal correspondence collapsed because acoustic features of the swapped track do not align with the visual semantics of the on-screen subject.")

    # Cleanup temp upload file if used
    if uploaded_temp_file and os.path.exists(uploaded_temp_file):
        try:
            os.remove(uploaded_temp_file)
        except Exception:
            pass

else:
    st.info("👆 Please select an AVE dataset sample or upload a video to inspect predictions.")
