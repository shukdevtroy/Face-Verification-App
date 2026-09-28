%%writefile app.py
"""
================================================================================
 FACE VERIFICATION APP — InsightFace (ArcFace) primary backend, AdaFace optional
================================================================================

WHY THIS IS MORE ACCURATE THAN A PLAIN DeepFace SETUP
--------------------------------------------------------------------------------
- InsightFace's "buffalo_l" pack uses a ResNet100 backbone trained with ArcFace
  loss on the Glint360K / WebFace600K datasets — larger and cleaner training
  data than what ships inside most DeepFace wrappers, so the embeddings are
  more discriminative out of the box.
- Its detector (SCRFD/RetinaFace) also returns 5 facial landmarks, which we use
  to do a proper similarity-transform alignment (not just a bounding-box crop),
  the same alignment convention ArcFace/AdaFace were trained on.
- AdaFace is included as a second, swappable backend because it was trained
  specifically to handle *low-quality* images (blurry scans, bad lighting,
  low-res webcams) better than plain ArcFace — useful for NID-vs-webcam checks.

INSTALL
--------------------------------------------------------------------------------
    pip install insightface onnxruntime gradio opencv-python-headless pillow numpy

    # Only needed if you also want the AdaFace backend — no manual repo
    # cloning or checkpoint downloading required, it auto-downloads from
    # Hugging Face on first run, same as InsightFace does:
    pip install torch transformers huggingface_hub

RUN
--------------------------------------------------------------------------------
    python face_verification_app.py

NOTE ON THRESHOLDS
--------------------------------------------------------------------------------
There is no single universally "correct" similarity threshold — it depends on
what false-accept vs. false-reject rate you're willing to tolerate for your
use case. The defaults below are reasonable starting points, but you should
tune them against your own labeled data (pairs of known-same and known-
different people) before relying on this for anything security-sensitive.
================================================================================
"""

import os
import sys
import numpy as np
import cv2
import gradio as gr
from PIL import Image

# ==============================================================================
# 1) INSIGHTFACE SETUP  (primary backend — detection + alignment + embedding)
# ==============================================================================
from insightface.app import FaceAnalysis
from insightface.utils import face_align  # gives us the standard 112x112 aligned crop

print("Loading InsightFace (buffalo_l) models... this can take a moment on first run "
      "(weights are downloaded automatically to ~/.insightface).")

# FaceAnalysis bundles a detector + landmark model + recognition model together.
# We build it ONCE at import time — loading models is slow, we don't want to
# reload them on every button click.
insight_app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
# ctx_id=-1 means CPU. If you have a CUDA GPU + onnxruntime-gpu installed,
# switch to ctx_id=0 for a big speed-up.
insight_app.prepare(ctx_id=-1, det_size=(640, 640))


# ==============================================================================
# 2) ADAFACE SETUP  (optional secondary backend — now auto-downloading)
# ==============================================================================
# AdaFace itself has no pip package, but its author (Minchul Kim) publishes the
# pretrained IR-101/WebFace12M checkpoint on Hugging Face Hub under the
# "CVLface" project (repo: minchul/cvlface_adaface_ir101_webface12m, MIT
# license). `transformers.AutoModel.from_pretrained(..., trust_remote_code=True)`
# downloads the weights AND a small wrapper.py that defines the model
# architecture, then loads it — no manual git clone or checkpoint hunting
# needed, the same way InsightFace auto-downloads buffalo_l above.
#
# SECURITY NOTE: trust_remote_code=True means Hugging Face will execute
# Python code shipped inside that model repo on your machine. It's coming
# from the paper's original author, but "trust_remote_code" always deserves
# a moment's thought before you point it at a repo you don't recognize.

ADAFACE_HF_REPO = "minchul/cvlface_adaface_ir101_webface12m"

ADAFACE_AVAILABLE = False
adaface_model = None


def try_load_adaface():
    """
    Downloads (on first run) and loads the AdaFace model from Hugging Face.
    Returns True on success, False otherwise. Failure is non-fatal — the app
    just hides the AdaFace option from the dropdown and runs InsightFace-only.
    """
    global ADAFACE_AVAILABLE, adaface_model
    try:
        import torch  # noqa: F401  (imported lazily so torch isn't a hard dependency)
        from transformers import AutoModel

        print(f"Loading AdaFace ({ADAFACE_HF_REPO}) from Hugging Face... "
              f"this downloads ~260MB the first time and caches it afterward.")

        model = AutoModel.from_pretrained(ADAFACE_HF_REPO, trust_remote_code=True)
        model.eval()

        adaface_model = model
        ADAFACE_AVAILABLE = True
        print("AdaFace backend loaded successfully.")
        return True
    except Exception as e:
        print(f"[Info] AdaFace backend unavailable ({e}). "
              f"Only InsightFace/ArcFace will be offered in the UI. "
              f"Make sure `torch` and `transformers` are installed.")
        return False


try_load_adaface()


def get_adaface_embedding(aligned_bgr_112):
    """
    Runs one 112x112 BGR aligned face crop through AdaFace and returns its
    embedding vector.

    Preprocessing here follows the CVLface model card's own quick-start
    example exactly: RGB image (note — RGB, not BGR, so we convert), pixel
    values scaled to [0, 1] then normalized with mean=0.5/std=0.5 per channel
    (equivalent to (pixel/255 - 0.5) / 0.5), giving roughly a [-1, 1] range.
    """
    import torch

    face_rgb = cv2.cvtColor(aligned_bgr_112, cv2.COLOR_BGR2RGB)
    face = face_rgb.astype(np.float32) / 255.0        # [0, 255] -> [0, 1]
    face = (face - 0.5) / 0.5                         # -> roughly [-1, 1]
    face = np.transpose(face, (2, 0, 1))              # HWC -> CHW
    face = np.expand_dims(face, axis=0)               # add batch dim -> (1, 3, 112, 112)
    tensor = torch.from_numpy(face).float()

    with torch.no_grad():
        embedding = adaface_model(tensor)  # CVLface wrapper returns the embedding directly

    return embedding.numpy().flatten()


# ==============================================================================
# 3) SHARED HELPER FUNCTIONS
# ==============================================================================

def rgb_to_bgr(img_rgb):
    """Gradio gives us RGB numpy arrays; OpenCV/InsightFace expect BGR."""
    return cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)


def bgr_to_rgb(img_bgr):
    """Convert back to RGB for display in Gradio."""
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)


def detect_largest_face(img_rgb):
    """
    Runs InsightFace's detector on an RGB image and returns the single
    largest detected face (in case multiple people appear in a photo).

    Returns:
        face (insightface Face object) or None if nothing was detected.
        img_bgr (the BGR version of the input, reused later for cropping)
    """
    img_bgr = rgb_to_bgr(img_rgb)
    faces = insight_app.get(img_bgr)  # list of detected Face objects
    if len(faces) == 0:
        return None, img_bgr

    # Pick the face with the largest bounding-box area — usually the
    # "main" subject in an ID photo or webcam frame.
    def bbox_area(f):
        x1, y1, x2, y2 = f.bbox
        return (x2 - x1) * (y2 - y1)

    largest = max(faces, key=bbox_area)
    return largest, img_bgr


def draw_bbox(img_rgb, face, label):
    """Draws a green rectangle + label around the detected face for display."""
    out = img_rgb.copy()
    x1, y1, x2, y2 = face.bbox.astype(int)
    cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 3)
    cv2.putText(out, label, (x1, max(0, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    return out


def align_face_112(img_bgr, face):
    """
    Uses the 5 facial landmarks (eyes, nose, mouth corners) to warp the face
    into the standard 112x112 alignment used by both ArcFace and AdaFace
    training pipelines. This is a similarity transform (rotate/scale/translate),
    not just a crop — it's what makes embeddings comparable across photos
    taken at different angles.
    """
    aligned_bgr = face_align.norm_crop(img_bgr, landmark=face.kps, image_size=112)
    return aligned_bgr


def cosine_similarity(a, b):
    """Standard cosine similarity between two embedding vectors."""
    a = a / (np.linalg.norm(a) + 1e-10)
    b = b / (np.linalg.norm(b) + 1e-10)
    return float(np.dot(a, b))


def embedding_heatmap(embedding, size=(300, 300)):
    """
    Purely for visualization: reshapes an embedding vector into a square
    grid and colorizes it, so you can visually compare two embeddings —
    similar faces should produce visually similar patterns.
    """
    n = len(embedding)
    grid_size = int(np.ceil(np.sqrt(n)))
    grid = np.zeros((grid_size, grid_size), dtype=np.float32)

    norm = (embedding - embedding.min()) / (embedding.max() - embedding.min() + 1e-10)
    for i in range(min(n, grid_size * grid_size)):
        grid[i // grid_size, i % grid_size] = norm[i]

    grid_img = cv2.resize(grid, size, interpolation=cv2.INTER_NEAREST)
    colored = cv2.applyColorMap((grid_img * 255).astype("uint8"), cv2.COLORMAP_JET)
    return cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)


# Reasonable starting-point thresholds for cosine similarity (higher = same person).
# TUNE THESE against your own data — see the module docstring above.
DEFAULT_THRESHOLDS = {
    "InsightFace (ArcFace, buffalo_l)": 0.36,
    "AdaFace (IR-101)": 0.30,
}


# ==============================================================================
# 4) MAIN VERIFICATION FUNCTION (wired to the Verify button)
# ==============================================================================

def verify_faces(nid_image, webcam_image, model_choice, threshold):
    """
    Full pipeline: detect -> align -> embed -> compare -> decide.
    Returns everything the UI needs to display, including intermediate steps.
    """
    empty = (None, None, None, None, None)  # placeholders for image outputs on error

    if nid_image is None or webcam_image is None:
        return "❌ Please upload both images.", *empty

    # ---- Step 1: Detect the largest face in each image ----------------------
    nid_face, nid_bgr = detect_largest_face(nid_image)
    if nid_face is None:
        return "❌ No face detected in the NID image.", *empty

    webcam_face, webcam_bgr = detect_largest_face(webcam_image)
    if webcam_face is None:
        return "❌ No face detected in the webcam image.", *empty

    nid_bbox_vis = draw_bbox(nid_image, nid_face, f"score={nid_face.det_score:.2f}")
    webcam_bbox_vis = draw_bbox(webcam_image, webcam_face, f"score={webcam_face.det_score:.2f}")

    # ---- Step 2: Align both faces to the standard 112x112 crop --------------
    nid_aligned = align_face_112(nid_bgr, nid_face)
    webcam_aligned = align_face_112(webcam_bgr, webcam_face)

    # ---- Step 3: Get embeddings using the chosen backend ---------------------
    if model_choice.startswith("InsightFace"):
        # InsightFace already computed a 512-d ArcFace embedding internally
        # during detect_largest_face() — no extra work needed.
        emb1 = nid_face.normed_embedding
        emb2 = webcam_face.normed_embedding
    else:  # AdaFace
        if not ADAFACE_AVAILABLE:
            return ("❌ AdaFace was selected but isn't loaded. Check that the "
                    "AdaFace repo + checkpoint paths at the top of this file are correct.",
                    *empty)
        emb1 = get_adaface_embedding(nid_aligned)
        emb2 = get_adaface_embedding(webcam_aligned)

    # ---- Step 4: Compare embeddings -------------------------------------------
    similarity = cosine_similarity(emb1, emb2)
    is_match = similarity >= threshold

    # ---- Step 5: Build visualizations for the intermediate steps --------------
    emb1_vis = embedding_heatmap(emb1)
    emb2_vis = embedding_heatmap(emb2)

    status = "✅ MATCH — same person" if is_match else "❌ NO MATCH — different people"
    color = "#1a7f37" if is_match else "#cf222e"

    summary_html = f"""
    <div style="padding:20px; border-radius:10px; border:2px solid {color}; background:#fafafa;">
        <h2 style="color:{color}; text-align:center; margin-top:0;">{status}</h2>
        <hr>
        <ul style="font-size:16px; line-height:1.7;">
            <li><b>Model used:</b> {model_choice}</li>
            <li><b>Cosine similarity:</b> {similarity:.4f}</li>
            <li><b>Threshold:</b> {threshold:.4f}</li>
            <li><b>Detection confidence (NID):</b> {nid_face.det_score:.4f}</li>
            <li><b>Detection confidence (Webcam):</b> {webcam_face.det_score:.4f}</li>
            <li><b>Embedding dimension:</b> {len(emb1)}D</li>
        </ul>
        <div style="background:#fff3cd; padding:10px; border-radius:6px;">
            💡 Higher cosine similarity means the two face embeddings point in a
            more similar direction in feature space — i.e., more likely the same
            person. Remember to tune the threshold on your own validation data.
        </div>
    </div>
    """

    return (
        summary_html,
        nid_bbox_vis,
        webcam_bbox_vis,
        bgr_to_rgb(nid_aligned),
        bgr_to_rgb(webcam_aligned),
    )


# ==============================================================================
# 5) GRADIO UI
# ==============================================================================

# Only offer AdaFace in the dropdown if it actually loaded successfully.
model_choices = ["InsightFace (ArcFace, buffalo_l)"]
if ADAFACE_AVAILABLE:
    model_choices.append("AdaFace (IR-101)")

with gr.Blocks(theme=gr.themes.Ocean()) as demo:

    gr.Markdown(
        """
        # 🔐 Face Verification — InsightFace (ArcFace) + AdaFace
        Upload an ID photo and a live/webcam photo to check whether they show
        the same person. Detection, alignment, and embedding comparison are
        all shown step by step below.
        """
    )

    with gr.Row():
        nid_input = gr.Image(label="📇 ID / NID Image", type="numpy", height=300)
        webcam_input = gr.Image(label="📷 Webcam / Live Image", type="numpy", height=300)

    with gr.Row():
        model_dropdown = gr.Dropdown(
            choices=model_choices,
            value=model_choices[0],
            label="🤖 Recognition Model",
        )
        threshold_slider = gr.Slider(
            minimum=0.0, maximum=1.0, step=0.01,
            value=DEFAULT_THRESHOLDS[model_choices[0]],
            label="Similarity Threshold (tune this for your data)",
        )

    # Keep the threshold slider's default in sync with whichever model is picked.
    def update_threshold(model_choice):
        return DEFAULT_THRESHOLDS.get(model_choice, 0.35)

    model_dropdown.change(fn=update_threshold, inputs=model_dropdown, outputs=threshold_slider)

    verify_button = gr.Button("🔍 Verify Faces", variant="primary", size="lg")

    result_output = gr.HTML(label="Result")

    gr.Markdown("### Step 1: Detected Faces")
    with gr.Row():
        nid_bbox_output = gr.Image(label="NID — Detected Face", type="numpy")
        webcam_bbox_output = gr.Image(label="Webcam — Detected Face", type="numpy")

    gr.Markdown("### Step 2: Aligned 112×112 Crops (fed into the recognition model)")
    with gr.Row():
        nid_aligned_output = gr.Image(label="NID — Aligned", type="numpy")
        webcam_aligned_output = gr.Image(label="Webcam — Aligned", type="numpy")

    verify_button.click(
        fn=verify_faces,
        inputs=[nid_input, webcam_input, model_dropdown, threshold_slider],
        outputs=[
            result_output,
            nid_bbox_output,
            webcam_bbox_output,
            nid_aligned_output,
            webcam_aligned_output,
        ],
    )

    gr.Markdown(
        """
        ---
        **Educational note:** this demonstrates a real production-style face
        verification pipeline. For any security-sensitive deployment, add
        liveness detection (to prevent someone verifying with a printed
        photo or a video replay), rate-limiting, and proper consent/privacy
        handling for biometric data.
        """
    )

if __name__ == "__main__":
    # demo.launch(share=True, debug=True)
    # For a private deployment instead of a public share link, use:
    demo.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860)), share=True)
