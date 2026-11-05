import html
import io
import json
from pathlib import Path

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

from knowledge_base import (
    HEALTHY_TIPS,
    MANGO_CLASSES,
    get_info,
    parse_class_name,
    quarantine_protocol,
)

st.set_page_config(
    page_title="Plant Disease Detection",
    page_icon="",
    layout="wide",
    initial_sidebar_state="collapsed",
)

HERE = Path(__file__).parent
MODEL_PATH = HERE / "trained_plant_disease_model.keras"
CLASS_NAMES_JSON = HERE / "class_names.json"  # optional, overrides the built-in list
IMG_SIZE = (128, 128)

# The model was trained on raw 0-255 pixels (no Rescaling layer, no /255).
# Keep this False unless you retrain with scaling.
NORMALIZE = False

# Palette
INK = "#12301f"
LEAF = "#2e7d32"
LEAF_SOFT = "#e3f0dc"
PAPER = "#f4f8f1"
MANGO = "#f2a900"
BLIGHT = "#b3261e"
BLIGHT_SOFT = "#fbe7e4"
AMBER = "#c77700"
MUTED = "#5d6f62"

# ----------------------------------------------------------------------------
# Styling
# ----------------------------------------------------------------------------
CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:opsz,wght@12..96,500;12..96,700;12..96,800&family=Source+Sans+3:wght@400;500;600&display=swap');

html, body, [class*="css"], .stApp {{
    font-family: 'Source Sans 3', system-ui, sans-serif;
    color: {INK};
}}
.stApp {{ background: {PAPER}; }}
[data-testid="stSidebar"], [data-testid="collapsedControl"], [data-testid="stSidebarCollapsedControl"],
header[data-testid="stHeader"], footer, #MainMenu {{ display: none !important; }}
.block-container {{ max-width: 1120px; padding-top: 1.4rem; padding-bottom: 4rem; }}

h1, h2, h3, h4 {{ font-family: 'Bricolage Grotesque', sans-serif; color: {INK}; letter-spacing: -0.01em; }}

/* Brand bar + nav */
.brand {{ display:flex; align-items:center; gap:10px; font-family:'Bricolage Grotesque',sans-serif;
         font-weight:800; font-size:22px; color:{INK}; padding-top:6px; }}
.brand .dot {{ width:14px; height:14px; border-radius:50% 50% 50% 0; background:{MANGO}; transform:rotate(-45deg); }}

div[role="radiogroup"] {{ gap: 6px; justify-content: flex-end; }}
div[role="radiogroup"] label {{ background: transparent; border: 1.5px solid transparent; border-radius: 999px;
    padding: 6px 18px; cursor: pointer; margin: 0; }}
div[role="radiogroup"] label > div:first-child {{ display: none; }}
div[role="radiogroup"] label p {{ font-weight: 600; font-size: 15px; color: {MUTED}; }}
div[role="radiogroup"] label:hover {{ background: {LEAF_SOFT}; }}
div[role="radiogroup"] label:has(input:checked) {{ background: {INK}; }}
div[role="radiogroup"] label:has(input:checked) p {{ color: #fff; }}
div[role="radiogroup"] label:focus-within {{ outline: 3px solid {MANGO}; outline-offset: 2px; }}

/* Hero */
.hero {{ padding: 56px 0 28px 0; max-width: 760px; }}
.hero h1 {{ font-size: clamp(38px, 6vw, 68px); line-height: 1.02; font-weight: 800; margin: 0 0 18px 0; }}
.hero p {{ font-size: 20px; line-height: 1.5; color: {MUTED}; margin: 0; max-width: 620px; }}

/* Steps */
.steps {{ display:grid; grid-template-columns: repeat(3, 1fr); gap: 0; margin: 36px 0 12px 0;
         border-top: 2px solid {INK}; }}
.step {{ padding: 18px 22px 8px 0; }}
.step .n {{ font-family:'Bricolage Grotesque',sans-serif; font-weight:800; font-size:15px; color:{LEAF}; }}
.step h4 {{ margin: 4px 0 6px 0; font-size: 20px; }}
.step p {{ margin: 0; color: {MUTED}; font-size: 16px; line-height: 1.45; }}

.section-title {{ font-size: 26px; font-weight: 700; margin: 40px 0 6px 0; }}
.chips {{ display:flex; flex-wrap:wrap; gap:8px; margin-top: 10px; }}
.chip {{ padding: 6px 14px; border-radius: 999px; font-size: 15px; font-weight: 500;
        background: {BLIGHT_SOFT}; color: {BLIGHT}; }}
.chip.ok {{ background: {LEAF_SOFT}; color: {LEAF}; }}

/* Primary buttons */
.stButton > button[kind="primary"], .stButton > button {{
    background: {MANGO}; color: {INK}; border: none; border-radius: 999px; font-weight: 700;
    padding: 0.55rem 1.6rem; font-size: 16px; }}
.stButton > button:hover {{ background: #ffbd1f; color: {INK}; }}
.stButton > button:focus-visible {{ outline: 3px solid {INK}; outline-offset: 2px; }}

/* Upload */
[data-testid="stFileUploader"] section {{ background: #fff; border: 2px dashed {LEAF}; border-radius: 14px; }}
[data-testid="stImage"] img {{ border-radius: 14px; }}

/* Verdict */
.verdict {{ border-radius: 18px; padding: 26px 28px; display:flex; gap: 26px; align-items:center; }}
.verdict.ok {{ background: {LEAF_SOFT}; }}
.verdict.bad {{ background: {BLIGHT_SOFT}; }}
.verdict .state {{ font-weight: 600; font-size: 15px; margin-bottom: 2px; }}
.verdict.ok .state, .verdict.ok .name {{ color: {LEAF}; }}
.verdict.bad .state, .verdict.bad .name {{ color: {BLIGHT}; }}
.verdict .name {{ font-family:'Bricolage Grotesque',sans-serif; font-weight: 800; font-size: 34px; line-height: 1.1; }}
.verdict .plant {{ color: {MUTED}; font-size: 16px; margin-top: 4px; }}

.ring {{ flex: 0 0 auto; width: 104px; height: 104px; border-radius: 50%; display:grid; place-items:center; }}
.ring .in {{ width: 78px; height: 78px; border-radius: 50%; background: #fff; display:flex;
            flex-direction:column; align-items:center; justify-content:center; }}
.ring .pct {{ font-family:'Bricolage Grotesque',sans-serif; font-weight:800; font-size:22px; line-height:1; }}
.ring .lbl {{ font-size: 12px; color: {MUTED}; margin-top: 2px; }}

/* Bars */
.bar-row {{ margin-bottom: 12px; }}
.bar-top {{ display:flex; justify-content:space-between; font-size:15px; margin-bottom:4px; }}
.bar-top span:last-child {{ font-weight:600; }}
.bar-track {{ background:#dfe8da; border-radius:6px; height:10px; overflow:hidden; }}
.bar-fill {{ height:10px; border-radius:6px; }}

.panel-title {{ font-family:'Bricolage Grotesque',sans-serif; font-weight:700; font-size:20px; margin: 0 0 12px 0; }}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {{ gap: 4px; border-bottom: 2px solid {INK}20; }}
.stTabs [data-baseweb="tab"] {{ font-weight: 600; padding: 10px 16px; }}
.stTabs [aria-selected="true"] {{ color: {INK}; }}
.stTabs [data-baseweb="tab-highlight"] {{ background: {MANGO}; height: 3px; }}

@media (max-width: 760px) {{
    .steps {{ grid-template-columns: 1fr; }}
    .verdict {{ flex-direction: column; align-items: flex-start; }}
}}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


# ----------------------------------------------------------------------------
# Model + classes
# ----------------------------------------------------------------------------
@st.cache_resource
def load_disease_model():
    return tf.keras.models.load_model(str(MODEL_PATH))


def resolve_class_names(n_out):
    """Pick the class list that matches the model's output size, or stop."""
    names = None
    if CLASS_NAMES_JSON.exists():
        names = json.loads(CLASS_NAMES_JSON.read_text(encoding="utf-8"))
    elif n_out == len(MANGO_CLASSES):
        names = MANGO_CLASSES

    if names is None or len(names) != n_out:
        st.error(
            f"The model has {n_out} outputs but the class list does not match "
            f"({'none found' if names is None else len(names)} names). "
            "Add a class_names.json that lists the classes in training order."
        )
        st.stop()
    return names


model = load_disease_model()
CLASS_NAMES = resolve_class_names(int(model.output_shape[-1]))


def predict_probs(img):
    """Resize exactly like training (tf bilinear) and return softmax probabilities."""
    arr = tf.image.resize(
        np.asarray(img, dtype=np.float32), IMG_SIZE, method="bilinear"
    ).numpy()
    if NORMALIZE:
        arr = arr / 255.0
    return model.predict(arr[None, ...], verbose=0)[0]


# ----------------------------------------------------------------------------
# UI helpers
# ----------------------------------------------------------------------------
def confidence_label(pct):
    if pct >= 85:
        return "High", LEAF
    if pct >= 60:
        return "Moderate", AMBER
    return "Low", BLIGHT


def esc(s):
    return html.escape(str(s))


def bar(pct, color, label):
    return (
        f"<div class='bar-row'><div class='bar-top'><span>{esc(label)}</span><span>{pct:.1f}%</span></div>"
        f"<div class='bar-track'><div class='bar-fill' style='width:{max(pct, 1):.1f}%;background:{color}'></div></div></div>"
    )


def verdict_card(plant, condition, healthy, conf, conf_lbl, conf_color):
    kind = "ok" if healthy else "bad"
    state = "Your leaf looks healthy" if healthy else "Disease detected"
    ring = (
        f"<div class='ring' style='background:conic-gradient({conf_color} {conf * 3.6:.1f}deg, #ffffff99 0)'>"
        f"<div class='in'><div class='pct' style='color:{conf_color}'>{conf:.0f}%</div>"
        f"<div class='lbl'>{conf_lbl}</div></div></div>"
    )
    st.markdown(
        f"<div class='verdict {kind}'>{ring}<div><div class='state'>{state}</div>"
        f"<div class='name'>{esc(condition)}</div><div class='plant'>{esc(plant)} leaf</div></div></div>",
        unsafe_allow_html=True,
    )


def go_to(page):
    st.session_state["page"] = page


# ----------------------------------------------------------------------------
# Result view
# ----------------------------------------------------------------------------
def render_result(probs):
    order = np.argsort(probs)[::-1]
    idx = int(order[0])
    class_name = CLASS_NAMES[idx]
    plant, condition, healthy = parse_class_name(class_name)
    info = get_info(class_name)
    conf = float(probs[idx]) * 100
    conf_lbl, conf_color = confidence_label(conf)

    verdict_card(plant, condition, healthy, conf, conf_lbl, conf_color)
    st.write("")

    if conf < 60:
        st.warning("The model is not sure. Retake the photo with one leaf, good light and a plain "
                   "background, or ask an agriculture expert.")
    elif len(order) > 1 and (float(probs[order[0]]) - float(probs[order[1]])) * 100 < 15:
        p2, c2, _ = parse_class_name(CLASS_NAMES[order[1]])
        st.info(f"Close call: the model is also considering **{p2}: {c2}**.")

    st.markdown("<div class='panel-title'>Other possibilities</div>", unsafe_allow_html=True)
    rows = ""
    for i in order[:3]:
        p, c, h = parse_class_name(CLASS_NAMES[i])
        rows += bar(float(probs[i]) * 100, LEAF if h else BLIGHT, f"{p}: {c}")
    st.markdown(rows, unsafe_allow_html=True)

    st.write("")
    tab_treat, tab_quar = st.tabs(["Treatment", "Quarantine"])

    with tab_treat:
        if healthy:
            st.success("No treatment needed. To keep it healthy:")
            for t in HEALTHY_TIPS:
                st.markdown(f"- {t}")
        else:
            if info.get("note"):
                st.info(info["note"])
            t_org, t_chem, t_prev = st.tabs(["Organic", "Chemical", "Prevention"])
            with t_org:
                for t in info["organic"]:
                    st.markdown(f"- {t}")
            with t_chem:
                for t in info["chemical"]:
                    st.markdown(f"- {t}")
                st.warning("Follow the product label, wear protective gear, and check that the product "
                           "is registered for your crop in your area. Respect the waiting period before "
                           "harvest. Ask a local agronomist before spraying.")
            with t_prev:
                for t in info["prevention"]:
                    st.markdown(f"- {t}")

    with tab_quar:
        if healthy:
            st.success("**No quarantine needed.**")
            st.markdown("- Check the plant every week.\n"
                        "- Keep new plants apart for about 2 weeks before placing them near this one.")
        else:
            proto = quarantine_protocol(info, None)
            box = {"high": st.error, "medium": st.warning, "low": st.info}[proto["urgency"]]
            box(f"**{proto['headline']}**")
            for n, step in enumerate(proto["steps"], 1):
                st.markdown(f"{n}. {step}")
            if proto["notify"]:
                st.error(proto["notify"])

    st.caption("This tool supports your decision but does not replace a plant pathologist or your local "
               "agriculture office. The model learned from leaf photos and may be less accurate on field "
               "photos with cluttered backgrounds.")


# ----------------------------------------------------------------------------
# Header + navigation
# ----------------------------------------------------------------------------
PAGES = ["Home", "Diagnose", "About"]
if "page" not in st.session_state:
    st.session_state["page"] = "Home"

h_left, h_right = st.columns([1, 1.4])
with h_left:
    st.markdown("<div class='brand'><span class='dot'></span>Mango Leaf Doctor</div>", unsafe_allow_html=True)
with h_right:
    st.radio("Navigate", PAGES, key="page", horizontal=True, label_visibility="collapsed")

page = st.session_state["page"]

# ----------------------------------------------------------------------------
# Pages
# ----------------------------------------------------------------------------
if page == "Home":
    st.markdown(
        "<div class='hero'><h1>Know what is wrong with your mango leaf.</h1>"
        "<p>Upload one photo and get a diagnosis, how sure the model is, treatment options "
        "and what to do next to protect the rest of your orchard.</p></div>",
        unsafe_allow_html=True,
    )
    st.button("Diagnose a leaf", type="primary", on_click=go_to, args=("Diagnose",))

    st.markdown(
        "<div class='steps'>"
        "<div class='step'><div class='n'>1</div><h4>Photograph one leaf</h4>"
        "<p>Good daylight, plain background, leaf filling most of the frame.</p></div>"
        "<div class='step'><div class='n'>2</div><h4>Upload it</h4>"
        "<p>JPG or PNG. The analysis starts as soon as the photo is in.</p></div>"
        "<div class='step'><div class='n'>3</div><h4>Read the report</h4>"
        "<p>Diagnosis, confidence, organic and chemical treatment, and a quarantine plan.</p></div>"
        "</div>",
        unsafe_allow_html=True,
    )

    chips = ""
    for c in CLASS_NAMES:
        _, cond, h = parse_class_name(c)
        chips += f"<span class='chip{' ok' if h else ''}'>{esc(cond)}</span>"
    st.markdown(
        f"<div class='section-title'>What it can recognise</div>"
        f"<p style='color:{MUTED};margin:0'>{len(CLASS_NAMES)} conditions, including healthy leaves.</p>"
        f"<div class='chips'>{chips}</div>",
        unsafe_allow_html=True,
    )

elif page == "About":
    st.markdown("<div class='section-title' style='margin-top:32px'>About this tool</div>", unsafe_allow_html=True)
    st.markdown(
        f"The model sorts mango leaf photos into **{len(CLASS_NAMES)} classes**. Treatment and quarantine "
        "advice is general guidance. For a confirmed diagnosis, consult a plant pathologist or your local "
        "agriculture office."
    )
    chips = ""
    for c in CLASS_NAMES:
        p, cond, h = parse_class_name(c)
        chips += f"<span class='chip{' ok' if h else ''}'>{esc(p)}: {esc(cond)}</span>"
    st.markdown(f"<div class='chips'>{chips}</div>", unsafe_allow_html=True)

else:  # Diagnose
    st.markdown("<div class='section-title' style='margin-top:28px'>Diagnose a leaf</div>", unsafe_allow_html=True)
    left, right = st.columns([1, 1.25], gap="large")

    with left:
        uploaded = st.file_uploader("Upload a mango leaf photo", type=["jpg", "jpeg", "png"])
        img = None
        if uploaded is not None:
            data = uploaded.getvalue()
            key = (uploaded.name, len(data))
            img = Image.open(io.BytesIO(data)).convert("RGB")
            st.image(img, caption=uploaded.name, use_container_width=True)

    with right:
        if img is None:
            st.info("Upload a photo to see the diagnosis here.")
        else:
            cached = st.session_state.get("result")
            if not cached or cached["key"] != key:
                with st.spinner("Analysing leaf..."):
                    probs = predict_probs(img)
                st.session_state["result"] = {"key": key, "probs": probs}
            render_result(st.session_state["result"]["probs"])