import streamlit as st
import google.generativeai as genai
import base64
import json
import io
import time
import numpy as np
from PIL import Image, ImageFilter, ImageDraw
import cv2
import tempfile
import os
from dotenv import load_dotenv
import google.generativeai as genai
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import textwrap

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=api_key)
model = genai.GenerativeModel("gemini-2.5-flash")

# ─── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="NailDx AI",
    page_icon="💅",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;700&display=swap');

:root {
    --bg-primary: #0a0d14;
    --bg-secondary: #111622;
    --bg-card: #161d2e;
    --bg-card-hover: #1a2235;
    --accent-blue: #4f8ef7;
    --accent-cyan: #00d4ff;
    --accent-green: #00e676;
    --accent-orange: #ff9800;
    --accent-red: #ff4444;
    --accent-purple: #9c6bff;
    --text-primary: #e8eaf0;
    --text-secondary: #8892a4;
    --border: #1e2d45;
    --border-glow: rgba(79, 142, 247, 0.3);
}

.stApp {
    background: var(--bg-primary) !important;
    font-family: 'Space Grotesk', sans-serif !important;
    color: var(--text-primary) !important;
}

.main .block-container {
    padding: 1.5rem 2rem !important;
    max-width: 1400px !important;
}

#MainMenu, footer, header { visibility: hidden !important; }
.stDeployButton { display: none !important; }

.nail-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 1rem 1.5rem;
    background: var(--bg-secondary);
    border-radius: 16px;
    border: 1px solid var(--border);
    margin-bottom: 1.5rem;
}
.nail-logo { display: flex; align-items: center; gap: 12px; }
.nail-logo-icon {
    width: 44px; height: 44px;
    background: linear-gradient(135deg, #4f8ef7, #9c6bff);
    border-radius: 12px;
    display: flex; align-items: center; justify-content: center;
    font-size: 22px;
}
.nail-logo-text h1 {
    margin: 0; font-size: 1.4rem; font-weight: 700;
    background: linear-gradient(90deg, #4f8ef7, #00d4ff);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}
.nail-logo-text p { margin: 0; font-size: 0.75rem; color: var(--text-secondary); }

section[data-testid="stSidebar"] {
    background: var(--bg-secondary) !important;
    border-right: 1px solid var(--border) !important;
}
section[data-testid="stSidebar"] .stMarkdown h2 {
    color: var(--text-secondary) !important;
    font-size: 0.7rem !important;
    text-transform: uppercase !important;
    letter-spacing: 2px !important;
    margin-bottom: 0.5rem !important;
}

.card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 1.25rem;
    margin-bottom: 1rem;
}
.card-title {
    font-size: 0.85rem;
    font-weight: 600;
    color: var(--text-secondary);
    text-transform: uppercase;
    letter-spacing: 1.5px;
    margin-bottom: 1rem;
}

.result-condition {
    background: linear-gradient(135deg, rgba(79,142,247,0.15), rgba(156,107,255,0.1));
    border: 1px solid rgba(79,142,247,0.3);
    border-radius: 12px;
    padding: 1rem 1.2rem;
    margin-bottom: 0.75rem;
}
.result-condition .label { font-size: 0.7rem; color: var(--text-secondary); margin-bottom: 4px; }
.result-condition .value { font-size: 1.4rem; font-weight: 700; color: var(--text-primary); }
.result-condition .sub { font-size: 0.8rem; color: var(--text-secondary); }

.metric-box {
    background: var(--bg-secondary);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 0.85rem 1rem;
    text-align: center;
}
.metric-box .m-label { font-size: 0.7rem; color: var(--text-secondary); margin-bottom: 6px; }
.metric-box .m-value { font-size: 1.6rem; font-weight: 700; }
.metric-box .m-badge {
    display: inline-block;
    font-size: 0.65rem; font-weight: 600;
    padding: 2px 8px; border-radius: 20px;
    margin-top: 4px;
}

.badge-high { background: rgba(255,68,68,0.2); color: #ff6b6b; }
.badge-medium { background: rgba(255,152,0,0.2); color: #ffb74d; }
.badge-low { background: rgba(0,230,118,0.2); color: #69f0ae; }

.severity-bar-bg {
    background: var(--border);
    height: 8px; border-radius: 4px;
    overflow: hidden; margin: 8px 0 4px;
}
.severity-bar-fill {
    height: 100%; border-radius: 4px;
    transition: width 0.8s ease;
}

.finding-item {
    display: flex; align-items: flex-start; gap: 10px;
    padding: 8px 0;
    border-bottom: 1px solid var(--border);
    font-size: 0.875rem; color: var(--text-primary);
}
.finding-item:last-child { border-bottom: none; }
.finding-check { color: var(--accent-green); font-size: 1rem; flex-shrink: 0; margin-top: 1px; }

.rec-item {
    display: flex; align-items: flex-start; gap: 10px;
    padding: 8px 0;
    border-bottom: 1px solid var(--border);
    font-size: 0.875rem; color: var(--text-primary);
}
.rec-item:last-child { border-bottom: none; }
.rec-icon { font-size: 1rem; flex-shrink: 0; margin-top: 1px; }

.history-item {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 0.75rem;
    margin-bottom: 0.5rem;
    display: flex; justify-content: space-between; align-items: center;
}
.history-score {
    width: 32px; height: 32px;
    border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    font-size: 0.75rem; font-weight: 700;
}

[data-testid="stFileUploader"] {
    background: var(--bg-card) !important;
    border: 2px dashed var(--border) !important;
    border-radius: 16px !important;
    padding: 1rem !important;
}

.stButton > button {
    background: linear-gradient(135deg, #4f8ef7, #9c6bff) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    font-family: 'Space Grotesk', sans-serif !important;
    font-weight: 600 !important;
    padding: 0.6rem 1.2rem !important;
    transition: all 0.2s !important;
}
.stButton > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 20px rgba(79,142,247,0.3) !important;
}

.stTabs [data-baseweb="tab-list"] {
    background: var(--bg-secondary) !important;
    border-radius: 10px !important;
    padding: 4px !important;
    gap: 4px !important;
}
.stTabs [data-baseweb="tab"] {
    background: transparent !important;
    color: var(--text-secondary) !important;
    border-radius: 8px !important;
    font-family: 'Space Grotesk', sans-serif !important;
}
.stTabs [aria-selected="true"] {
    background: var(--bg-card) !important;
    color: var(--text-primary) !important;
}

.analyzing-text {
    text-align: center;
    color: var(--accent-cyan);
    font-size: 0.9rem;
    font-family: 'JetBrains Mono', monospace;
    margin-top: 0.5rem;
}

.info-bar {
    background: var(--bg-secondary);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 0.6rem 1rem;
    text-align: center;
    font-size: 0.75rem;
    color: var(--text-secondary);
    margin-top: 1rem;
}

hr { border-color: var(--border) !important; }

.stSelectbox > div > div {
    background: var(--bg-card) !important;
    border-color: var(--border) !important;
    color: var(--text-primary) !important;
}

.stRadio > div { flex-direction: row !important; gap: 1rem !important; }

/* Camera button style */
[data-testid="stCameraInput"] button {
    background-color: #4f8ef7 !important;   /* button color */
    color: white !important;                /* text color */
    border-radius: 10px !important;
    font-weight: 600 !important;
    border: none !important;
}

/* Hover effect */
[data-testid="stCameraInput"] button:hover {
    background-color: #3b78d8 !important;
}

/* Nutrition card */
/* Nutrition card */
.nutrition-card {
    background: #0F1A2B;
    border: 1px solid #00B7C2;
    border-radius: 12px;
    padding: 1rem;
    margin-bottom: 0.75rem;
    display: flex;
    align-items: flex-start;
    gap: 12px;
}

/* Icon */
.nutrition-icon {
    font-size: 2rem;
    flex-shrink: 0;
}

/* Title (Biotin, Iron, Zinc etc) */
.nutrition-body h4 {
    margin: 0 0 4px 0;
    font-size: 0.95rem;
    font-weight: 600;
    color: #FFFFFF;
}

/* Description text */
.nutrition-body p {
    margin: 0;
    font-size: 0.8rem;
    color: #9FB3C8;
    line-height: 1.5;
}

/* Food sources text */
.nutrition-sources {
    margin-top: 8px;
    font-size: 0.75rem;
    color: #00C8FF;
}


/* Live detection overlay */
.detection-status-good {
    background: rgba(0,230,118,0.1);
    border: 1px solid rgba(0,230,118,0.4);
    border-radius: 10px;
    padding: 10px 14px;
    font-size: 0.85rem;
    color: #00e676;
    margin-bottom: 10px;
}
.detection-status-warn {
    background: rgba(255,152,0,0.1);
    border: 1px solid rgba(255,152,0,0.4);
    border-radius: 10px;
    padding: 10px 14px;
    font-size: 0.85rem;
    color: #ffb74d;
    margin-bottom: 10px;
}
.detection-status-bad {
    background: rgba(255,68,68,0.1);
    border: 1px solid rgba(255,68,68,0.4);
    border-radius: 10px;
    padding: 10px 14px;
    font-size: 0.85rem;
    color: #ff6b6b;
    margin-bottom: 10px;
}
/* AI generated chat text */
[data-testid="stChatMessageContent"] {
    color: #e8eaf0 !important;
}

/* bullet points inside AI text */
[data-testid="stChatMessageContent"] li {
    color: #e8eaf0 !important;
}

/* paragraphs inside AI text */
[data-testid="stChatMessageContent"] p {
    color: #e8eaf0 !important;
}
</style>
""", unsafe_allow_html=True)


# ─── Helpers ───────────────────────────────────────────────────────────────────

def image_to_base64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    return base64.b64encode(buf.getvalue()).decode()


def generate_heatmap(img: Image.Image) -> Image.Image:
    img_cv = cv2.cvtColor(np.array(img.convert("RGB")), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (15, 15), 0)
    edges = cv2.Canny(blurred, 30, 100)
    edges = cv2.GaussianBlur(edges.astype(float), (31, 31), 0)
    hsv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2HSV)
    sat = hsv[:, :, 1].astype(float)
    combined = (edges * 0.5 + sat * 0.5)
    combined = cv2.GaussianBlur(combined, (51, 51), 0)
    if combined.max() > 0:
        combined = (combined / combined.max() * 255).astype(np.uint8)
    heatmap = cv2.applyColorMap(combined, cv2.COLORMAP_JET)
    heatmap_rgb = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    orig = np.array(img.convert("RGB").resize((img_cv.shape[1], img_cv.shape[0])))
    blended = cv2.addWeighted(orig, 0.35, heatmap_rgb, 0.65, 0)
    mask = (combined > 120).astype(np.uint8)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest = max(contours, key=cv2.contourArea)
        cv2.drawContours(blended, [largest], -1, (255, 255, 255), 2)
    return Image.fromarray(blended)


def detect_nail_live(frame_rgb: np.ndarray) -> tuple:
    """
    Robust nail detection using multi-channel skin + nail plate analysis.
    Returns (is_nail: bool, confidence_pct: int, feedback: str, details: dict)
    """
    h, w = frame_rgb.shape[:2]
    score = 0
    details = {}

    # ── 1. Convert colour spaces ────────────────────────────────────────────────
    hsv  = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2HSV)
    ycrcb = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2YCrCb)
    gray  = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)

    # ── 2. Skin detection — HSV + YCrCb dual-range ────────────────────────────
    # HSV skin ranges (light to dark skin tones)
    skin_hsv_ranges = [
        (np.array([0,  15,  60], np.uint8), np.array([25, 200, 255], np.uint8)),   # warm skin
        (np.array([0,  10,  80], np.uint8), np.array([20, 180, 255], np.uint8)),   # light skin
        (np.array([155,15,  60], np.uint8), np.array([180,200, 255], np.uint8)),   # pinkish/wrap-around hue
    ]
    skin_mask = np.zeros((h, w), dtype=np.uint8)
    for lo, hi in skin_hsv_ranges:
        skin_mask = cv2.bitwise_or(skin_mask, cv2.inRange(hsv, lo, hi))

    # YCrCb skin range (robust across lighting)
    ycrcb_mask = cv2.inRange(ycrcb,
                              np.array([0,  133, 77],  np.uint8),
                              np.array([255,173, 127], np.uint8))
    combined_skin = cv2.bitwise_and(skin_mask, ycrcb_mask)

    # Morphological cleanup
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    combined_skin = cv2.morphologyEx(combined_skin, cv2.MORPH_CLOSE, kernel)
    combined_skin = cv2.morphologyEx(combined_skin, cv2.MORPH_OPEN,  cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5)))

    skin_ratio = np.sum(combined_skin > 0) / (h * w)
    details["skin_ratio"] = round(skin_ratio, 3)

    if skin_ratio > 0.20:
        score += 25
    elif skin_ratio > 0.10:
        score += 15
    elif skin_ratio > 0.04:
        score += 5

    # ── 3. Nail plate detection — bright, low-saturation region ───────────────
    # Nail plate is typically brighter and less saturated than surrounding skin
    sat_ch  = hsv[:, :, 1].astype(float)
    val_ch  = hsv[:, :, 2].astype(float)

    # Low saturation (nail plate is pinkish-white, not vivid)
    low_sat_mask  = (sat_ch < 90).astype(np.uint8) * 255
    # Moderate-to-bright value
    bright_mask   = (val_ch > 130).astype(np.uint8) * 255

    nail_plate_mask = cv2.bitwise_and(low_sat_mask, bright_mask)
    # Must overlap with skin region
    nail_plate_mask = cv2.bitwise_and(nail_plate_mask, combined_skin)

    nail_ratio = np.sum(nail_plate_mask > 0) / (h * w)
    details["nail_ratio"] = round(nail_ratio, 3)

    # Focus bonus: nail should be near the centre of the frame
    cy1, cy2 = int(h * 0.2), int(h * 0.8)
    cx1, cx2 = int(w * 0.2), int(w * 0.8)
    centre_region = nail_plate_mask[cy1:cy2, cx1:cx2]
    centre_nail_ratio = np.sum(centre_region > 0) / ((cy2 - cy1) * (cx2 - cx1) + 1e-6)
    details["centre_nail_ratio"] = round(centre_nail_ratio, 3)

    if nail_ratio > 0.08:
        score += 25
    elif nail_ratio > 0.03:
        score += 15
    elif nail_ratio > 0.01:
        score += 5

    if centre_nail_ratio > 0.10:
        score += 15
    elif centre_nail_ratio > 0.04:
        score += 8

    # ── 4. Shape analysis — look for oval/rectangular nail contour ────────────
    nail_clean = cv2.morphologyEx(nail_plate_mask, cv2.MORPH_CLOSE,
                                   cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)))
    contours, _ = cv2.findContours(nail_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    has_nail_shape = False
    best_area = 0
    if contours:
        largest = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest)
        best_area = area
        min_area = h * w * 0.015   # nail must cover ≥1.5% of frame
        if area > min_area:
            x, y, cw, ch_ = cv2.boundingRect(largest)
            aspect = cw / ch_ if ch_ > 0 else 0
            # Typical fingernail aspect ratio: 0.4 (narrow portrait) to 2.5 (wide landscape)
            if 0.35 <= aspect <= 2.8:
                # Convexity check: nails are convex / nearly convex
                hull_area = cv2.contourArea(cv2.convexHull(largest))
                if hull_area > 0:
                    solidity = area / hull_area
                    if solidity > 0.55:   # nails are fairly solid shapes
                        has_nail_shape = True

    details["has_nail_shape"] = has_nail_shape
    details["best_contour_area"] = round(best_area)

    if has_nail_shape:
        score += 20

    # ── 5. Edge density — nail boundary should have clear edges ────────────────
    blurred_gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred_gray, 40, 120)
    # Only consider edges inside the skin area
    edge_in_skin = cv2.bitwise_and(edges, combined_skin)
    edge_density = np.sum(edge_in_skin > 0) / (skin_ratio * h * w + 1e-6)
    details["edge_density"] = round(edge_density, 3)

    if 0.05 < edge_density < 0.55:   # moderate edges = nail boundary present
        score += 10

    # ── 6. Brightness check — nail should be reasonably lit ────────────────────
    mean_brightness = float(np.mean(gray))
    details["brightness"] = round(mean_brightness)
    if mean_brightness < 50:
        score -= 15   # too dark
    elif mean_brightness > 230:
        score -= 10   # over-exposed

    # ── 7. Blurriness check ────────────────────────────────────────────────────
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    details["sharpness"] = round(laplacian_var)
    if laplacian_var < 80:
        score -= 10   # too blurry

    score = max(0, min(100, score))

    # ── 8. Decision + feedback ─────────────────────────────────────────────────
    if score >= 65:
        return True, score, "✅ Nail detected clearly — ready to analyze!", details
    elif score >= 40:
        if skin_ratio < 0.10:
            msg = "⚠️ Move the camera closer — nail should fill the frame."
        elif not has_nail_shape:
            msg = "⚠️ Nail detected but shape unclear — adjust angle or lighting."
        elif mean_brightness < 50:
            msg = "⚠️ Image is too dark — move to better lighting."
        elif laplacian_var < 80:
            msg = "⚠️ Image is blurry — hold the camera steady."
        else:
            msg = "⚠️ Partially detected — centre the nail and move closer."
        return False, score, msg, details
    else:
        if skin_ratio < 0.05:
            msg = "❌ No finger/nail detected. Point the camera directly at your fingernail."
        elif mean_brightness < 50:
            msg = "❌ Too dark to detect. Use brighter lighting."
        elif laplacian_var < 80:
            msg = "❌ Image too blurry. Hold the camera very still."
        else:
            msg = "❌ Nail not detected. Ensure the nail fills most of the frame."
        return False, score, msg, details


def analyze_nail_with_ai(img: Image.Image) -> dict:
    prompt = """
You are NailDx AI, a nail health analysis system. Analyze this nail image and return ONLY valid JSON with no markdown, no explanation, just JSON.

{
  "condition": "Name of nail condition or Normal",
  "condition_code": "ICD-10 or short code if applicable",
  "confidence": 85,
  "risk_score": 4,
  "risk_level": "Low|Medium|High",
  "severity": "Mild|Moderate|Severe|None",
  "severity_pct": 40,
  "clinical_findings": ["finding 1", "finding 2", "finding 3"],
  "recommendations": ["rec 1", "rec 2", "rec 3", "rec 4"],
  "nutrition_links": ["Biotin", "Iron", "Zinc"],
  "summary": "2-3 sentence clinical summary"
}
"""
    response = model.generate_content([prompt, img])
    raw = response.text.strip()
    raw = raw.replace("```json", "").replace("```", "").strip()
    try:
        return json.loads(raw)
    except:
        return {
            "condition": "Analysis Error",
            "condition_code": "",
            "confidence": 0,
            "risk_score": 0,
            "risk_level": "Unknown",
            "severity": "Unknown",
            "severity_pct": 0,
            "clinical_findings": ["AI response format error"],
            "recommendations": ["Try scanning again"],
            "nutrition_links": [],
            "summary": raw
        }


# ─── Session State Init ─────────────────────────────────────────────────────────
defaults = {
    "scan_history": [],
    "current_result": None,
    "current_image": None,
    "heatmap_image": None,
    "active_nav": "Scan & Analyze",
    "chat_messages": [],
    "risk_history": [],
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ─── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="padding: 0.5rem 0 1rem;">
        <div style="display:flex;align-items:center;gap:10px;margin-bottom:1.5rem;">
            <div style="width:40px;height:40px;background:linear-gradient(135deg,#4f8ef7,#9c6bff);
                        border-radius:10px;display:flex;align-items:center;justify-content:center;font-size:20px;">💅</div>
            <div>
                <div style="font-weight:700;font-size:1.1rem;background:linear-gradient(90deg,#4f8ef7,#00d4ff);
                            -webkit-background-clip:text;-webkit-text-fill-color:transparent;">NailDx AI</div>
                <div style="font-size:0.7rem;color:#8892a4;">AI-powered nail health analysis</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    nav_items = [
        ("🔬", "Scan & Analyze"),
        ("💬", "AI Follow-up Chat"),
        ("🥗", "Nutrition Insights"),
    ]

    st.markdown("## NAVIGATION")
    for icon, label in nav_items:
        active = st.session_state.active_nav == label
        if st.button(f"{icon}  {label}", key=f"nav_{label}", use_container_width=True):
            st.session_state.active_nav = label
            st.rerun()

    st.markdown("---")
    st.markdown("## SCAN HISTORY")
    if st.session_state.scan_history:
        for h in reversed(st.session_state.scan_history[-5:]):
            score = h["risk_score"]
            color = "#ff6b6b" if score >= 7 else "#ffb74d" if score >= 4 else "#69f0ae"
            st.markdown(f"""
            <div class="history-item">
                <div>
                    <div style="font-size:0.8rem;font-weight:600;">{h['date']}</div>
                    <div style="font-size:0.72rem;color:#8892a4;">{h['condition']}</div>
                </div>
                <div class="history-score" style="background:rgba(255,255,255,0.05);color:{color};">
                    {score}
                </div>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.markdown('<p style="color:#8892a4;font-size:0.8rem;">No scans yet.</p>', unsafe_allow_html=True)


# ─── Main Content ───────────────────────────────────────────────────────────────

# Header
col_h1, col_h2 = st.columns([3, 1])
with col_h1:
    st.markdown("""
    <div class="nail-header">
        <div class="nail-logo">
            <div class="nail-logo-icon">💅</div>
            <div class="nail-logo-text">
                <h1>NailDx AI</h1>
                <p>AI-powered nail health analysis</p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

with col_h2:
    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("🔄 New Scan", use_container_width=True):
        st.session_state.current_result = None
        st.session_state.current_image = None
        st.session_state.heatmap_image = None
        st.session_state.active_nav = "Scan & Analyze"
        st.rerun()


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: Scan & Analyze
# ═══════════════════════════════════════════════════════════════════════════════
if st.session_state.active_nav == "Scan & Analyze":

    col_mode1, col_mode2, col_mode3 = st.columns([1, 2, 1])
    with col_mode2:
        input_mode = st.radio(
            "Choose Input Method",
            ["📁 Upload / Drag & Drop", "📷 Live Camera"],
            horizontal=True,
            label_visibility="collapsed"
        )

    uploaded_img = None

    if input_mode == "📁 Upload / Drag & Drop":
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">📁 Upload Nail Image</div>', unsafe_allow_html=True)
        uploaded_file = st.file_uploader(
            "Drag & drop or click to upload",
            type=["jpg", "jpeg", "png", "bmp", "webp"],
            label_visibility="collapsed"
        )
        if uploaded_file:
            uploaded_img = Image.open(uploaded_file).convert("RGB")
        st.markdown('</div>', unsafe_allow_html=True)

    else:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">📷 Live Camera — Point at your nail</div>', unsafe_allow_html=True)
        st.markdown("""
        <div style="background:rgba(79,142,247,0.08);border:1px solid rgba(79,142,247,0.2);
                    border-radius:10px;padding:10px 14px;margin-bottom:10px;font-size:0.8rem;color:#8892a4;">
            💡 <strong style="color:#e8eaf0;">Tips for best detection:</strong><br>
            • Hold your fingernail <strong>5–10 cm</strong> from the camera<br>
            • The nail should fill <strong>at least 50%</strong> of the frame<br>
            • Use <strong>bright natural or white light</strong> — avoid shadows<br>
            • Keep the camera <strong>steady</strong> to avoid blur<br>
            • Remove nail polish for accurate health analysis
        </div>
        """, unsafe_allow_html=True)

        camera_frame = st.camera_input("Take a photo of your nail", label_visibility="collapsed")

        if camera_frame:
            cam_img = Image.open(camera_frame).convert("RGB")
            frame_arr = np.array(cam_img)

            is_nail, confidence_score, feedback_msg, det_details = detect_nail_live(frame_arr)

            # ── Detection result bar ────────────────────────────────────────────
            bar_color = "#00e676" if is_nail else "#ff9800" if confidence_score >= 40 else "#ff4444"
            bar_pct   = confidence_score

            st.markdown(f"""
                <div style="background:rgba(0,0,0,0.3);border:1px solid #1e2d45;border-radius:12px;padding:12px 16px;margin-bottom:10px;">
                    <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:6px;">
                        <span style="font-size:0.85rem;font-weight:600;color:#e8eaf0;">{feedback_msg}</span>
                        <span style="font-size:0.75rem;color:#8892a4;">Detection: <strong style="color:{bar_color};">{confidence_score}%</strong></span>
                    </div>
                    <div style="background:#1e2d45;height:6px;border-radius:3px;overflow:hidden;">
                        <div style="width:{bar_pct}%;height:100%;background:{bar_color};border-radius:3px;transition:width 0.5s;"></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

            if is_nail:
                uploaded_img = cam_img
            else:
                # Show diagnostic details to help user reposition
                brightness = det_details.get("brightness", 0)
                sharpness  = det_details.get("sharpness", 0)
                skin_pct   = round(det_details.get("skin_ratio", 0) * 100, 1)
                nail_pct   = round(det_details.get("nail_ratio", 0) * 100, 1)

                bright_icon  = "✅" if brightness > 80  else "⚠️"
                sharp_icon   = "✅" if sharpness > 80   else "⚠️"
                skin_icon    = "✅" if skin_pct > 15    else "⚠️"
                nail_icon    = "✅" if nail_pct > 3     else "⚠️"
                shape_icon   = "✅" if det_details.get("has_nail_shape") else "⚠️"

                st.markdown(f"""
                <div style="background:rgba(255,152,0,0.06);border:1px solid rgba(255,152,0,0.2);
                            border-radius:10px;padding:12px 16px;margin-bottom:10px;font-size:0.8rem;">
                    <div style="font-weight:600;color:#ffb74d;margin-bottom:8px;">📋 Detection Diagnostics</div>
                    <div style="display:grid;grid-template-columns:1fr 1fr;gap:6px;color:#8892a4;">
                        <div>{bright_icon} Brightness: <strong style="color:#e8eaf0;">{brightness}/255</strong></div>
                        <div>{sharp_icon} Sharpness: <strong style="color:#e8eaf0;">{sharpness:.0f}</strong></div>
                        <div>{skin_icon} Skin coverage: <strong style="color:#e8eaf0;">{skin_pct}%</strong></div>
                        <div>{nail_icon} Nail plate: <strong style="color:#e8eaf0;">{nail_pct}%</strong></div>
                        <div>{shape_icon} Nail shape: <strong style="color:#e8eaf0;">{'Found' if det_details.get('has_nail_shape') else 'Not found'}</strong></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                col_o1, col_o2, col_o3 = st.columns([2, 1, 2])
                with col_o2:
                    if st.button("Analyze Anyway", use_container_width=True):
                        uploaded_img = cam_img

        st.markdown('</div>', unsafe_allow_html=True)

    # ─── Analyze Button ─────────────────────────────────────────────────────────
    if uploaded_img:
        prev_c1, prev_c2, prev_c3 = st.columns([1, 1, 1])
        with prev_c2:
            st.image(uploaded_img, caption="📸 Preview — Ready to analyze", use_container_width=True)

        col_btn1, col_btn2, col_btn3 = st.columns([2, 1, 2])
        with col_btn2:
            analyze_clicked = st.button("🔬 Analyze Nail", use_container_width=True)

        if analyze_clicked:
            with st.spinner(""):
                placeholder = st.empty()
                steps = [
                    "🔍 Preprocessing image...",
                    "🧠 Running AI vision analysis...",
                    "🗺️ Generating attention heatmap...",
                    "📋 Compiling clinical findings...",
                    "✅ Analysis complete!"
                ]
                for step in steps:
                    placeholder.markdown(f'<div class="analyzing-text">{step}</div>', unsafe_allow_html=True)
                    time.sleep(0.6)

                try:
                    result = analyze_nail_with_ai(uploaded_img)
                    heatmap = generate_heatmap(uploaded_img)

                    st.session_state.current_result = result
                    st.session_state.current_image = uploaded_img
                    st.session_state.heatmap_image = heatmap
                    st.session_state.chat_messages = []

                    st.session_state.scan_history.append({
                        "date": datetime.now().strftime("%b %d, %Y %I:%M %p"),
                        "date_obj": datetime.now().isoformat(),
                        "condition": result.get("condition", "Unknown"),
                        "risk_score": result.get("risk_score", 0),
                        "confidence": result.get("confidence", 0),
                        "severity": result.get("severity", "Unknown"),
                    })
                    st.session_state.risk_history.append({
                        "datetime": datetime.now().isoformat(),
                        "score": result.get("risk_score", 0),
                        "condition": result.get("condition", "Unknown"),
                    })
                    placeholder.empty()
                    st.rerun()
                except Exception as e:
                    placeholder.empty()
                    st.error(f"❌ Error: {str(e)}")

    # ─── Results Display ─────────────────────────────────────────────────────────
    if st.session_state.current_result and st.session_state.current_image:
        r = st.session_state.current_result
        risk_score = r.get("risk_score", 0)
        confidence = r.get("confidence", 0)
        risk_level = r.get("risk_level", "Low")
        severity_pct = r.get("severity_pct", 40)

        risk_color = "#ff4444" if risk_level == "High" else "#ff9800" if risk_level == "Medium" else "#00e676"
        badge_class = "badge-high" if risk_level == "High" else "badge-medium" if risk_level == "Medium" else "badge-low"

        st.markdown("---")

        c1, c2, c3 = st.columns([1.2, 1.2, 1])

        with c1:
            st.markdown('<div class="card"><div class="card-title">📸 Uploaded Nail Image</div>', unsafe_allow_html=True)
            st.image(st.session_state.current_image, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        with c2:
            st.markdown('<div class="card"><div class="card-title">🗺️ AI Heatmap (Regions of Concern)</div>', unsafe_allow_html=True)
            st.image(st.session_state.heatmap_image, use_container_width=True)
            st.markdown("""
            <div style="display:flex;align-items:center;gap:8px;margin-top:6px;">
                <span style="font-size:0.7rem;color:#8892a4;">Low</span>
                <div style="flex:1;height:6px;border-radius:3px;
                    background:linear-gradient(to right,#0000ff,#00ffff,#00ff00,#ffff00,#ff0000);"></div>
                <span style="font-size:0.7rem;color:#8892a4;">High Concern</span>
            </div>
            """, unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

        with c3:
            st.markdown('<div class="card"><div class="card-title">📊 Analysis Results</div>', unsafe_allow_html=True)
            st.markdown(f"""
            <div class="result-condition">
                <div class="label">🔬 Condition</div>
                <div class="value">{r.get('condition', 'Unknown')}</div>
                <div class="sub">({r.get('condition_code', '')})</div>
            </div>
            """, unsafe_allow_html=True)

            m1, m2 = st.columns(2)
            with m1:
                st.markdown(f"""
                <div class="metric-box">
                    <div class="m-label">👥 Confidence</div>
                    <div class="m-value" style="color:#4f8ef7;">{confidence}%</div>
                </div>
                """, unsafe_allow_html=True)
            with m2:
                st.markdown(f"""
                <div class="metric-box">
                    <div class="m-label">🛡️ Risk Score</div>
                    <div class="m-value" style="color:{risk_color};">{risk_score}<span style="font-size:0.9rem;color:#8892a4;">/10</span></div>
                    <div class="m-badge {badge_class}">{risk_level} Risk</div>
                </div>
                """, unsafe_allow_html=True)

            st.markdown(f"""
            <div style="margin-top:12px;">
                <div style="display:flex;justify-content:space-between;align-items:center;">
                    <span style="font-size:0.8rem;color:#8892a4;">Severity</span>
                </div>
                <div class="severity-bar-bg">
                    <div class="severity-bar-fill" style="width:{severity_pct}%;
                        background:linear-gradient(90deg,{risk_color}88,{risk_color});"></div>
                </div>
                <div style="font-size:0.78rem;color:#e8eaf0;margin-top:2px;">{r.get('severity','Moderate')}</div>
            </div>
            """, unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

        c4, c5 = st.columns(2)
        with c4:
            findings = r.get("clinical_findings", [])
            findings_html = "".join([
                f'<div class="finding-item"><span class="finding-check">✅</span><span>{f}</span></div>'
                for f in findings
            ])
            st.markdown(f"""
            <div class="card">
                <div class="card-title">🩺 Clinical Findings</div>
                {findings_html}
            </div>
            """, unsafe_allow_html=True)

        with c5:
            recs = r.get("recommendations", [])
            rec_icons = ["🏥", "🧴", "💊", "💅", "🌿", "🔬"]
            recs_html = "".join([
                f'<div class="rec-item"><span class="rec-icon">{rec_icons[i % len(rec_icons)]}</span><span>{rec}</span></div>'
                for i, rec in enumerate(recs)
            ])
            st.markdown(f"""
            <div class="card">
                <div class="card-title">💡 Recommendations</div>
                {recs_html}
            </div>
            """, unsafe_allow_html=True)

        st.markdown(f"""
        <div class="card">
            <div class="card-title">📝 Clinical Summary</div>
            <p style="color:var(--text-primary);font-size:0.9rem;line-height:1.6;margin:0;">{r.get('summary', '')}</p>
        </div>
        """, unsafe_allow_html=True)

    elif not uploaded_img:
        st.markdown("""
        <div style="text-align:center;padding:4rem 2rem;">
            <div style="font-size:4rem;margin-bottom:1rem;">💅</div>
            <div style="font-size:1.2rem;font-weight:600;margin-bottom:0.5rem;">Upload or Capture a Nail Image</div>
            <div style="color:#8892a4;font-size:0.875rem;max-width:400px;margin:0 auto;">
                Use the input method above to upload a nail photo or take one with your camera.
                Our AI will analyze it for health conditions.
            </div>
        </div>
        """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: AI Follow-up Chat
# ═══════════════════════════════════════════════════════════════════════════════
elif st.session_state.active_nav == "AI Follow-up Chat":
    st.markdown('<div class="card"><div class="card-title">💬 AI Follow-up Chat</div>', unsafe_allow_html=True)

    if not st.session_state.current_result:
        st.markdown("""
        <div style="text-align:center;padding:3rem;">
            <div style="font-size:3rem;margin-bottom:1rem;">💬</div>
            <div style="font-weight:600;font-size:1rem;margin-bottom:0.5rem;">No Scan Result Available</div>
            <div style="color:#8892a4;font-size:0.875rem;">
                Please run a nail scan first from the <strong>Scan &amp; Analyze</strong> page,
                then return here to chat with the AI about your results.
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        r = st.session_state.current_result
        st.markdown(f"""
        <div style="background:rgba(79,142,247,0.08);border:1px solid rgba(79,142,247,0.2);
                    border-radius:10px;padding:10px 14px;margin-bottom:12px;font-size:0.8rem;">
            💡 Chatting about: <strong style="color:#4f8ef7;">{r.get('condition','Unknown')}</strong>
            &nbsp;·&nbsp; Risk: <strong>{r.get('risk_score',0)}/10</strong>
            &nbsp;·&nbsp; Confidence: <strong>{r.get('confidence',0)}%</strong>
        </div>
        """, unsafe_allow_html=True)

        if not st.session_state.chat_messages:
            st.markdown('<div style="font-size:0.8rem;color:#8892a4;margin-bottom:8px;">💡 Suggested questions:</div>', unsafe_allow_html=True)
            sugg_cols = st.columns(3)
            suggestions = [
                "What causes this condition?",
                "Is this serious?",
                "What should I eat to improve nail health?",
                "When should I see a doctor?",
                "How long does it take to heal?",
                "Can nail polish make it worse?",
            ]
            for i, suggestion in enumerate(suggestions):
                with sugg_cols[i % 3]:
                    if st.button(suggestion, key=f"sugg_{i}", use_container_width=True):
                        st.session_state.chat_messages.append({"role": "user", "content": suggestion})
                        context = json.dumps(st.session_state.current_result, indent=2)
                        with st.spinner("Thinking..."):
                            response = model.generate_content(
                                f"""You are NailDx AI assistant. The user's nail scan result: {context}
User question: {suggestion}
Answer clearly and concisely as a helpful medical AI assistant. Keep response under 200 words."""
                            )
                            st.session_state.chat_messages.append({"role": "assistant", "content": response.text})
                        st.rerun()

        for msg in st.session_state.chat_messages:
            with st.chat_message(msg["role"]):
                st.write(msg["content"])

        if prompt_text := st.chat_input("Ask about your nail health..."):
            st.session_state.chat_messages.append({"role": "user", "content": prompt_text})
            with st.chat_message("user"):
                st.write(prompt_text)

            context = json.dumps(st.session_state.current_result, indent=2)
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    response = model.generate_content(
                        f"""You are NailDx AI assistant. The user's nail scan result: {context}
User question: {prompt_text}
Answer clearly and concisely as a helpful medical AI assistant."""
                    )
                    answer = response.text
                    st.write(answer)
                    st.session_state.chat_messages.append({"role": "assistant", "content": answer})

        if st.session_state.chat_messages:
            if st.button("🗑️ Clear Chat"):
                st.session_state.chat_messages = []
                st.rerun()

    st.markdown('</div>', unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: Nutrition Insights
# ═══════════════════════════════════════════════════════════════════════════════
elif st.session_state.active_nav == "Nutrition Insights":
    st.markdown('<div class="card"><div class="card-title">🥗 Nutrition Insights for Nail Health</div>', unsafe_allow_html=True)

    if st.session_state.current_result:
        nuts = st.session_state.current_result.get("nutrition_links", [])
        if nuts:
            st.markdown(f"""
            <div style="background:rgba(0,212,255,0.08);border:1px solid rgba(0,212,255,0.2);
                        border-radius:10px;padding:10px 14px;margin-bottom:16px;font-size:0.85rem;">
                🎯 <strong style="color:#00d4ff;">Personalized for your scan:</strong>
                AI identified links to: <strong>{', '.join(nuts)}</strong>
            </div>
            """, unsafe_allow_html=True)

    nutrition_data = [
        {
            "icon": "🥚",
            "name": "Biotin (Vitamin B7)",
            "description": "Biotin is essential for keratin production — the protein that makes up your nails. Deficiency leads to brittle, thin nails that break easily.",
            "sources": "Eggs, almonds, sweet potato, spinach, salmon",
            "daily": "30–100 mcg/day",
            "tag": "Biotin"
        },
        {
            "icon": "🥩",
            "name": "Iron",
            "description": "Iron deficiency causes koilonychia (spoon-shaped nails), pale nail beds, and slow nail growth. Vital for oxygen transport to nail matrix.",
            "sources": "Red meat, lentils, spinach, tofu, fortified cereals",
            "daily": "8–18 mg/day",
            "tag": "Iron"
        },
        {
            "icon": "🦪",
            "name": "Zinc",
            "description": "Zinc supports nail plate formation and prevents white spots (leukonychia). Deficiency causes slow growth, ridges, and brittleness.",
            "sources": "Oysters, pumpkin seeds, beef, chickpeas, cashews",
            "daily": "8–11 mg/day",
            "tag": "Zinc"
        },
        {
            "icon": "🥛",
            "name": "Calcium",
            "description": "Though nails are made of keratin (not calcium), adequate calcium supports overall nail strength and prevents softening.",
            "sources": "Milk, yogurt, cheese, broccoli, almonds, tofu",
            "daily": "1000–1200 mg/day",
            "tag": "Calcium"
        },
        {
            "icon": "🐟",
            "name": "Omega-3 Fatty Acids",
            "description": "Omega-3s hydrate the nail plate and cuticle, preventing dryness and brittleness. They reduce inflammation around the nail fold.",
            "sources": "Salmon, mackerel, flaxseed, walnuts, chia seeds",
            "daily": "1.1–1.6 g/day",
            "tag": "Omega-3"
        },
        {
            "icon": "🫐",
            "name": "Vitamin C",
            "description": "Vitamin C is needed for collagen synthesis which supports the nail bed. Deficiency can cause nail fragility and slow healing.",
            "sources": "Bell peppers, citrus, strawberries, kiwi, broccoli",
            "daily": "65–90 mg/day",
            "tag": "Vitamin C"
        },
        {
            "icon": "🥜",
            "name": "Protein",
            "description": "Nails are 90% keratin — a protein. Inadequate protein leads to slow growth, soft, brittle nails and Beau's lines.",
            "sources": "Chicken, fish, eggs, legumes, dairy, tofu",
            "daily": "0.8 g per kg body weight",
            "tag": "Protein"
        },
        {
            "icon": "🌞",
            "name": "Vitamin D",
            "description": "Vitamin D regulates calcium absorption and immune function. Deficiency is linked to nail psoriasis and fragile nails.",
            "sources": "Sunlight, fatty fish, egg yolks, fortified milk",
            "daily": "600–800 IU/day",
            "tag": "Vitamin D"
        },
    ]

    scan_nuts = st.session_state.current_result.get("nutrition_links", []) if st.session_state.current_result else []

    for n in nutrition_data:
        highlight = n["tag"] in scan_nuts
        border = "border: 1px solid rgba(0,212,255,0.4);" if highlight else ""
        badge = '<span style="background:rgba(0,212,255,0.15);color:#00d4ff;border-radius:20px;padding:2px 8px;font-size:0.65rem;font-weight:600;margin-left:8px;">🎯 LINKED TO YOUR SCAN</span>' if highlight else ""
        st.markdown(f"""
        <div class="nutrition-card" style="{border}">
            <div class="nutrition-icon">{n['icon']}</div>
            <div class="nutrition-body">
                <h4>{n['name']} {badge}</h4>
                <p>{n['description']}</p>
                <div class="nutrition-sources">
                    🍽️ <strong>Food sources:</strong> {n['sources']}<br>
                    💊 <strong>Daily intake:</strong> {n['daily']}
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)


# ─── Info Bar ───────────────────────────────────────────────────────────────────
st.markdown("""
<div class="info-bar">
    ℹ️ For informational screening only — not a substitute for professional medical advice.
    Always consult a dermatologist or healthcare provider for diagnosis and treatment.
</div>
""", unsafe_allow_html=True)