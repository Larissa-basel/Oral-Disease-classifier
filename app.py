import os
import json

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

from tensorflow.keras.models import load_model
from tensorflow.keras.applications.efficientnet import preprocess_input as effnet_preprocess
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess

SAVE_DIR = "saved_models"

MODEL_DISPLAY_NAMES = {
    "custom_cnn": "Custom CNN",
    "efficientnetb0": "EfficientNetB0",
    "resnet50": "ResNet50",
}

st.set_page_config(page_title="Oral Disease Classifier — Model Comparison", page_icon="🦷", layout="wide")


@st.cache_data
def load_metadata():
    metadata_path = os.path.join(SAVE_DIR, "metadata.json")
    class_path = os.path.join(SAVE_DIR, "class_indices.json")

    print("Metadata path:", os.path.abspath(metadata_path))
    print("Class path:", os.path.abspath(class_path))

    with open(metadata_path) as f:
        metadata = json.load(f)

    print("Metadata contents:")
    print(metadata)

    with open(class_path) as f:
        raw = json.load(f)

    idx_to_class = {int(k): v for k, v in raw["idx_to_class"].items()}

    return metadata, idx_to_class


@st.cache_resource
def load_models(metadata):
    models = {}
    for name, info in metadata["models"].items():
        models[name] = load_model(os.path.join(SAVE_DIR, info["file"]), safe_mode=False)
    return models


def preprocess_image(img, kind, img_size):
    img = img.convert("RGB").resize((img_size, img_size))
    arr = np.array(img).astype("float32")
    if kind == "rescale_1_255":
        arr = arr / 255.0
    elif kind == "efficientnet":
        arr = effnet_preprocess(arr)
    elif kind == "resnet50":
        arr = resnet_preprocess(arr)
    return np.expand_dims(arr, axis=0)


def main():
    st.title("🦷 Oral Disease Classifier")
    st.caption(
        "Upload a photo and compare predictions across three models: a custom CNN, "
        "EfficientNetB0, and ResNet50."
    )

    if not os.path.isdir(SAVE_DIR):
        st.error(
            f"Couldn't find the '{SAVE_DIR}/' folder next to this app. Put metadata.json, "
            f"class_indices.json, and the three .keras files inside a folder named "
            f"'{SAVE_DIR}' in the same directory as app.py."
        )
        return

    metadata, idx_to_class = load_metadata()
    models = load_models(metadata)
    img_size = metadata["img_size"]
    class_names = [idx_to_class[i] for i in range(len(idx_to_class))]

    with st.expander("ℹ️ About these models (test-set performance)"):
        perf_rows = [
            {
                "Model": MODEL_DISPLAY_NAMES.get(key, key),
                "Test Accuracy": f"{info['test_accuracy'] * 100:.1f}%",
                "Macro F1": f"{info['macro_f1']:.3f}",
            }
            for key, info in metadata["models"].items()
        ]
        st.dataframe(pd.DataFrame(perf_rows), hide_index=True, use_container_width=True)

    uploaded_file = st.file_uploader("Upload an oral photo (JPG/PNG)", type=["jpg", "jpeg", "png"])

    if uploaded_file is None:
        st.info("Upload an image above to see predictions from all three models.")
        return

    image = Image.open(uploaded_file)

    col_img, col_summary = st.columns([1, 2])
    with col_img:
        st.image(image, caption="Uploaded image", use_container_width=True)

    predictions = {}
    prob_table = {}

    with st.spinner("Running all three models..."):
        for key, info in metadata["models"].items():
            x = preprocess_image(image, info["preprocessing"], img_size)
            probs = models[key].predict(x, verbose=0)[0]
            pred_idx = int(np.argmax(probs))
            predictions[key] = {
                "class": idx_to_class[pred_idx],
                "confidence": float(probs[pred_idx]),
            }
            prob_table[MODEL_DISPLAY_NAMES.get(key, key)] = probs

    with col_summary:
        st.subheader("Predictions")
        cols = st.columns(len(predictions))
        for col, (key, pred) in zip(cols, predictions.items()):
            col.metric(
                label=MODEL_DISPLAY_NAMES.get(key, key),
                value=pred["class"],
                delta=f"{pred['confidence'] * 100:.1f}% confidence",
                delta_color="off",
            )

        pred_classes = {p["class"] for p in predictions.values()}
        if len(pred_classes) == 1:
            st.success(f"All three models agree: **{pred_classes.pop()}**")
        else:
            st.warning("The models disagree on this one — see the full breakdown below.")

    st.divider()

    st.subheader("Full probability breakdown")
    df = pd.DataFrame(prob_table, index=class_names)
    df_pct = (df * 100).round(2)
    st.dataframe(df_pct.style.highlight_max(axis=0, color="#005667"), use_container_width=True)
    st.bar_chart(df_pct)

    st.subheader("Where the models disagree most")
    spread = ((df.max(axis=1) - df.min(axis=1)) * 100).round(2)
    spread_df = pd.DataFrame(
        {"Class": class_names, "Max - min probability (points)": spread.values}
    ).sort_values("Max - min probability (points)", ascending=False)
    st.dataframe(spread_df, hide_index=True, use_container_width=True)
    st.caption("A larger gap here means the models disagree more on how likely that class is for this image.")


if __name__ == "__main__":
    main()