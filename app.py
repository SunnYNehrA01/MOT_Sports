import streamlit as st
import tempfile
import os
import subprocess

# --- Page Configuration ---
st.set_page_config(
    page_title="MOT Sports AI", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom Professional Styling ---
st.markdown("""
    <style>
    /* Global Theme */
    .stApp {
        background-color: #0b0e11;
        color: #ffffff;
    }

    /* Bento Box Design */
    .bento-card {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 15px;
        padding: 1.5rem;
        margin-bottom: 1.5rem;
    }

    /* Professional Emerald Buttons */
    .stButton>button {
        width: 100%;
        border-radius: 8px;
        height: 3em;
        background-color: #00ff88 !important;
        color: #0b0e11 !important;
        font-weight: 700;
        border: none;
    }
    .stButton>button:hover {
        background-color: #00cc6d !important;
    }

    /* Comparison Table Styling */
    .comparison-table {
        width: 100%;
        border-collapse: collapse;
        margin-top: 10px;
        font-size: 0.82rem;
    }
    .comparison-table th {
        text-align: left;
        color: #00ff88;
        border-bottom: 1px solid rgba(255,255,255,0.1);
        padding: 8px 6px;
        background: rgba(0, 255, 136, 0.06);
    }
    .comparison-table td {
        padding: 8px 6px;
        border-bottom: 1px solid rgba(255,255,255,0.05);
        vertical-align: top;
    }
    .comparison-table tr:hover td {
        background: rgba(255, 255, 255, 0.03);
    }

    .metric-chip {
        display: inline-block;
        border: 1px solid rgba(0, 255, 136, 0.35);
        border-radius: 999px;
        padding: 2px 10px;
        margin: 2px 6px 2px 0;
        font-size: 0.72rem;
        color: #9fffd2;
        background: rgba(0, 255, 136, 0.08);
    }

    .sidebar-note {
        border-left: 3px solid #00ff88;
        padding: 0.65rem 0.8rem;
        background: rgba(255, 255, 255, 0.03);
        border-radius: 8px;
        margin-top: 0.5rem;
        margin-bottom: 0.5rem;
    }

    [data-testid="stSidebar"] {
        background-color: #0d1117;
    }
    
    h1, h2, h3 { color: #ffffff !important; }
    .stCaption { color: rgba(255, 255, 255, 0.5); }
    </style>
    """, unsafe_allow_html=True)

def convert_to_h264(input_path, output_path):
    """Converts OpenCV output to browser-compatible H.264.
    Returns True when conversion succeeds, False otherwise.
    """
    command = [
        "ffmpeg",
        "-y",
        "-i",
        input_path,
        "-c:v",
        "libx264",
        "-crf",
        "23",
        "-preset",
        "fast",
        "-pix_fmt",
        "yuv420p",
        output_path,
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    return result.returncode == 0 and os.path.exists(output_path) and os.path.getsize(output_path) > 0

def main():
    if 'processed' not in st.session_state:
        st.session_state.processed = False
    if 'output_bytes' not in st.session_state:
        st.session_state.output_bytes = None
    if 'output_filename' not in st.session_state:
        st.session_state.output_filename = None

    # --- VIEW 1: Upload and Configuration ---
    if not st.session_state.processed:
        _, center_col, _ = st.columns([1, 4, 1])

        with center_col:
            st.title("MOT Sports Tracker")
            st.markdown("Advanced Identity Persistence and Tactical Analysis")
            k1, k2, k3 = st.columns(3)
            with k1:
                st.markdown('<span class="metric-chip">Multi-Object Tracking</span>', unsafe_allow_html=True)
            with k2:
                st.markdown('<span class="metric-chip">Visual Re-Identification</span>', unsafe_allow_html=True)
            with k3:
                st.markdown('<span class="metric-chip">Camera Motion Compensation</span>', unsafe_allow_html=True)
            st.divider()

            with st.sidebar:
                st.header("Pipeline Configuration")
                config_tab, guide_tab = st.tabs(["⚙️ Config", "📊 Model Guide"])

                with config_tab:
                    sport_type = st.selectbox(
                        "Sport Category",
                        ["Basketball", "Football", "Cricket", "Racing", "General"],
                        help="Adjusts tracker assumptions and post-processing behavior for sport context.",
                    )

                    st.subheader("Model Selection")
                    model_size = st.select_slider(
                        "Architecture",
                        options=["yolo11n.pt", "yolo11s.pt", "yolo11m.pt"],
                        value="yolo11s.pt",
                        help="Choose a model based on your performance vs. precision needs.",
                    )

                    with st.expander("Advanced Parameters"):
                        use_reid = st.checkbox("Enable Visual Re-ID", value=True)
                        use_cmc = st.checkbox("Camera Motion Compensation", value=True)
                        conf_thresh = st.slider("Confidence Threshold", 0.1, 0.9, 0.25)

                with guide_tab:
                    st.markdown("""
                    <div class="sidebar-note">
                        <strong>Quick Tip:</strong> For most laptop GPUs/CPUs, <strong>yolo11s.pt</strong> is the best default.
                        Use <strong>yolo11m.pt</strong> when player overlap and long-range visibility are frequent.
                    </div>
                    """, unsafe_allow_html=True)

                    st.markdown("""
                    <table class="comparison-table">
                        <tr>
                            <th>Model</th>
                            <th>Params</th>
                            <th>Speed</th>
                            <th>Accuracy</th>
                            <th>Best For</th>
                            <th>Trade-off</th>
                        </tr>
                        <tr>
                            <td><strong>Nano (n)</strong><br/>yolo11n.pt</td>
                            <td>2.6M</td>
                            <td>Very Fast</td>
                            <td>Standard</td>
                            <td>Real-time inference, edge devices, quick iteration</td>
                            <td>Lower small-object consistency in crowded frames</td>
                        </tr>
                        <tr>
                            <td><strong>Small (s)</strong><br/>yolo11s.pt</td>
                            <td>9.4M</td>
                            <td>Balanced</td>
                            <td>High</td>
                            <td>General sports analysis and stable ID persistence</td>
                            <td>Moderate compute demand</td>
                        </tr>
                        <tr>
                            <td><strong>Medium (m)</strong><br/>yolo11m.pt</td>
                            <td>20.1M</td>
                            <td>Moderate</td>
                            <td>Maximum</td>
                            <td>Dense scenes, tactical review, highest detection fidelity</td>
                            <td>Higher latency and larger memory footprint</td>
                        </tr>
                    </table>
                    """, unsafe_allow_html=True)

                    st.caption("Model parameter values are approximate and intended for quick planning.")

            st.markdown('<div class="bento-card">', unsafe_allow_html=True)
            uploaded_file = st.file_uploader("Upload video file (MP4, MOV, AVI)", type=["mp4", "mov", "avi"])
            st.markdown('</div>', unsafe_allow_html=True)

            if uploaded_file:
                st.markdown('<div class="bento-card">', unsafe_allow_html=True)
                st.write("Processing Information")
                st.caption(f"File: {uploaded_file.name} | Model: {model_size}")
                
                if st.button("Run Tracking Engine"):
                    try:
                        from core.engine import TrackingEngine
                    except ImportError as e:
                        st.error(
                            "Tracking dependencies are unavailable in this environment. "
                            "Please install system OpenCV requirements or use the headless OpenCV package."
                        )
                        st.exception(e)
                        st.stop()

                    tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
                    tfile.write(uploaded_file.read())
                    input_path = tfile.name
                    tfile.close()

                    args = {
                        'model': model_size, 'use_reid': use_reid, 'use_cmc': use_cmc,
                        'conf': conf_thresh, 'high_thresh': 0.6, 'low_thresh': 0.1, 'max_lost': 60
                    }
                    
                    engine = TrackingEngine(args)
                    raw_fd, raw_out = tempfile.mkstemp(suffix=".mp4")
                    web_fd, web_out = tempfile.mkstemp(suffix=".mp4")
                    os.close(raw_fd)
                    os.close(web_fd)

                    p_bar = st.progress(0)
                    try:
                        with st.spinner("Analyzing frames and resolving identities..."):
                            engine.process_video(input_path, raw_out, sport_type, lambda p: p_bar.progress(p))
                        
                        convert_to_h264(raw_out, web_out)

                        with open(web_out, "rb") as video_file:
                            st.session_state.output_bytes = video_file.read()
                        st.session_state.output_filename = f"analysis_{os.path.splitext(uploaded_file.name)[0]}.mp4"
                        st.session_state.processed = True
                        st.rerun()

                    except Exception as e:
                        st.error(f"Processing Error: {e}")
                    finally:
                        if os.path.exists(raw_out):
                            os.remove(raw_out)
                        if os.path.exists(web_out):
                            os.remove(web_out)
                        if os.path.exists(input_path):
                            os.remove(input_path)
                st.markdown('</div>', unsafe_allow_html=True)

    # --- VIEW 2: Results View ---
    else:
        _, res_col, _ = st.columns([1, 6, 1])

        with res_col:
            st.title("Analysis Complete")
            st.caption("Temporal Consistency and ID Persistence Active")
            st.divider()

            col_vid, col_meta = st.columns([3, 2])

            with col_vid:
                st.markdown('<div class="bento-card">', unsafe_allow_html=True)
                st.video(st.session_state.output_bytes)
                st.markdown('</div>', unsafe_allow_html=True)

            with col_meta:
                st.markdown('<div class="bento-card">', unsafe_allow_html=True)
                st.write("Export Metadata")
                st.info("The output video is encoded in H.264 for full browser and player support.")
                
                st.download_button(
                    label="Download Analysis Video",
                    data=st.session_state.output_bytes,
                    file_name=st.session_state.output_filename or "analysis.mp4",
                    mime="video/mp4"
                )
                st.markdown('</div>', unsafe_allow_html=True)

                if st.button("New Project"):
                    st.session_state.processed = False
                    st.session_state.output_bytes = None
                    st.session_state.output_filename = None
                    st.rerun()

if __name__ == "__main__":
    main()
