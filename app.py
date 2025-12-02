"""
NeuroSight - Brain Tumor MRI Classification Application
Streamlit web interface for classifying brain MRI scans using a custom CNN
"""

import streamlit as st
import numpy as np
from PIL import Image
from pathlib import Path
import random
import os
import time
import re
from datetime import datetime
from dotenv import load_dotenv
import plotly.express as px
import plotly.graph_objects as go
from prediction import predict_single, predict_batch, LABEL_MAP

load_dotenv()
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")

TEST_DATA_PATH = Path(__file__).parent / "data" / "brain-tumor-mri-dataset" / "Testing"

TUMOR_COLORS = {
    'notumor': '#1d3827',
    'meningioma': '#1c2d43',
    'pituitary': '#3d3b12',
    'glioma': '#3c2427'
}

def colored_box(tumor_type: str, content: str):
    """Display a colored box for tumor type"""
    color = TUMOR_COLORS.get(tumor_type, '#333333')
    st.markdown(
        f"""
        <div style="background-color: {color}; padding: 1rem; border-radius: 0.5rem; color: white; margin-bottom: 1rem;">
        {content}
        </div>
        """,
        unsafe_allow_html=True
    )

def get_random_test_images(n: int) -> list:
    """Get n random images from test set with their true labels"""
    all_images = []
    for class_name in ["notumor", "meningioma", "pituitary", "glioma"]:
        class_dir = TEST_DATA_PATH / class_name
        if class_dir.exists():
            for img_path in class_dir.glob("*.jpg"):
                all_images.append((img_path, class_name))

    selected = random.sample(all_images, min(n, len(all_images)))
    return selected

def create_stacked_bar(probabilities: dict):
    """Create a stacked bar chart showing class probabilities"""
    fig = go.Figure()
    for label, prob in sorted(probabilities.items(), key=lambda x: -x[1]):
        fig.add_trace(go.Bar(
            name=label,
            x=[prob],
            y=[''],
            orientation='h',
            marker_color=TUMOR_COLORS.get(label, '#888888'),
            text=f'{label}: {prob:.1%}' if prob > 0.05 else '',
            textposition='inside',
            hovertemplate=f'{label}: {prob:.1%}<extra></extra>'
        ))

    fig.update_layout(
        barmode='stack',
        height=80,
        margin=dict(l=0, r=0, t=0, b=0),
        showlegend=False,
        xaxis=dict(range=[0, 1], showticklabels=False, showgrid=False),
        yaxis=dict(showticklabels=False)
    )
    return fig

def create_pie_chart(data: dict, title: str):
    """Create a pie chart for data distribution"""
    fig = px.pie(
        values=list(data.values()),
        names=list(data.keys()),
        title=title,
        color=list(data.keys()),
        color_discrete_map=TUMOR_COLORS
    )
    fig.update_traces(textposition='inside', textinfo='percent+label')
    fig.update_layout(height=300, margin=dict(l=20, r=20, t=40, b=20))
    return fig

def load_training_history():
    """Load training history from JSON file"""
    import json
    history_path = Path(__file__).parent / "notebooks" / "exploration" / "brain_tumor_cnn_improved" / "training_history.json"
    if history_path.exists():
        with open(history_path, 'r') as f:
            return json.load(f)
    return None

def create_learning_curves(history: dict):
    """Create learning curve plots for accuracy and loss"""
    if not history:
        return None, None

    epochs = list(range(1, len(history['accuracy']) + 1))

    fig_acc = go.Figure()
    fig_acc.add_trace(go.Scatter(
        x=epochs, y=history['accuracy'],
        mode='lines+markers',
        name='Training Accuracy',
        line=dict(color='#2E86AB', width=3),
        marker=dict(size=8)
    ))
    fig_acc.add_trace(go.Scatter(
        x=epochs, y=history['val_accuracy'],
        mode='lines+markers',
        name='Validation Accuracy',
        line=dict(color='#A23B72', width=3, dash='dash'),
        marker=dict(size=8)
    ))
    fig_acc.update_layout(
        title='Model Accuracy Over Training',
        xaxis_title='Epoch',
        yaxis_title='Accuracy',
        hovermode='x unified',
        height=400,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    fig_loss = go.Figure()
    fig_loss.add_trace(go.Scatter(
        x=epochs, y=history['loss'],
        mode='lines+markers',
        name='Training Loss',
        line=dict(color='#2E86AB', width=3),
        marker=dict(size=8)
    ))
    fig_loss.add_trace(go.Scatter(
        x=epochs, y=history['val_loss'],
        mode='lines+markers',
        name='Validation Loss',
        line=dict(color='#A23B72', width=3, dash='dash'),
        marker=dict(size=8)
    ))
    fig_loss.update_layout(
        title='Model Loss Over Training',
        xaxis_title='Epoch',
        yaxis_title='Loss',
        hovermode='x unified',
        height=400,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    return fig_acc, fig_loss

def generate_pdf_report(report_text: str, images: list, file_names: list, results: list) -> bytes:
    """Generate a PDF report with images and classifications"""
    try:
        from fpdf import FPDF

        def clean_text(text):
            text = re.sub(r'[^\x00-\x7F]+', '', text)
            text = re.sub(r'#{1,6}\s', '', text)
            text = re.sub(r'\*\*(.+?)\*\*', r'\1', text)
            text = re.sub(r'\*(.+?)\*', r'\1', text)
            text = re.sub(r'>\s', '', text)
            text = re.sub(r'---+', '', text)
            text = re.sub(r'^-\s', '  - ', text, flags=re.MULTILINE)
            return text.strip()

        class PDF(FPDF):
            def header(self):
                self.set_font('Arial', 'B', 18)
                self.cell(0, 10, 'NeuroSight MRI Report', 0, 1, 'C')
                self.ln(5)

            def footer(self):
                self.set_y(-15)
                self.set_font('Arial', 'I', 8)
                self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')

        pdf = PDF()
        pdf.add_page()
        pdf.set_font("Arial", size=11)

        cleaned_report = clean_text(report_text)
        pdf.multi_cell(0, 5, cleaned_report)

        pdf.ln(10)
        pdf.set_font("Arial", 'B', 14)
        pdf.cell(0, 10, 'Image Classifications', 0, 1)
        pdf.set_font("Arial", size=10)

        for idx, (img, file_name, result) in enumerate(zip(images, file_names, results)):
            pdf.ln(5)
            pdf.set_font("Arial", 'B', 11)
            pdf.cell(0, 5, f"Scan {idx+1}: {file_name}", 0, 1)
            pdf.set_font("Arial", size=10)
            pdf.cell(0, 5, f"Classification: {result['label'].title()} (Confidence: {result['confidence']:.1%})", 0, 1)

            temp_img_path = f"/tmp/temp_img_{idx}_{int(time.time())}.jpg"
            img_pil = Image.fromarray(img)
            if img_pil.mode == 'RGBA':
                rgb_img = Image.new('RGB', img_pil.size, (255, 255, 255))
                rgb_img.paste(img_pil, mask=img_pil.split()[3])
                rgb_img.save(temp_img_path)
            else:
                img_pil.convert('RGB').save(temp_img_path)

            pdf.image(temp_img_path, x=10, w=100)

            try:
                os.remove(temp_img_path)
            except:
                pass

            pdf.ln(5)

        pdf_output = pdf.output(dest='S')
        if isinstance(pdf_output, bytes):
            return pdf_output
        else:
            return pdf_output.encode('latin-1')
    except ImportError:
        raise ImportError("PDF generation requires 'fpdf' library. Install with: pip install fpdf")
    except Exception as e:
        raise Exception(f"Error generating PDF: {str(e)}")

st.set_page_config(
    page_title="Brain Tumor Classification",
    page_icon="🧠",
    layout="wide"
)

st.markdown("""
    <style>
    .stToast {
        display: none !important;
    }
    audio {
        display: none !important;
    }
    </style>
    """, unsafe_allow_html=True)

st.title("NeuroSight - Brain Tumor MRI Classification")
st.markdown("Upload MRI brain scans to classify tumor types using a custom CNN.")

with st.sidebar:
    st.header("About")
    st.markdown("""
    **Brain Tumor Classification**

    This app uses a custom CNN model to classify brain MRI scans into:
    - **No Tumor** - healthy scan
    - **Meningioma** - usually benign
    - **Pituitary** - pituitary gland tumor
    - **Glioma** - malignant tumor

    For educational purposes only
    """)

tab1, tab2, tab3 = st.tabs(["Upload Images", "Test Model Performance", "About"])

with tab2:
    st.markdown("**Select random images from the test dataset**")

    num_images = st.slider("Number of random images to test", min_value=1, max_value=100, value=10)

    if st.button("Load Random Test Images", type="primary"):
        test_images = get_random_test_images(num_images)
        st.session_state['test_images'] = test_images

    if 'test_images' in st.session_state and st.session_state['test_images']:
        test_images = st.session_state['test_images']
        st.markdown(f"**{len(test_images)} {'image' if len(test_images) == 1 else 'images'} loaded**")

        with st.spinner("Processing images..."):
            images = []
            true_labels = []
            file_names = []
            for img_path, true_label in test_images:
                img = Image.open(img_path)
                img_array = np.array(img)
                images.append(img_array)
                true_labels.append(true_label)
                file_names.append(img_path.name)

            if len(images) == 1:
                results = [predict_single(images[0])]
            else:
                results = predict_batch(images)

        st.session_state['results'] = results
        st.session_state['true_labels'] = true_labels

        st.markdown("---")
        st.subheader("Results")

        cols_per_row = min(len(images), 3)

        for i in range(0, len(images), cols_per_row):
            cols = st.columns(cols_per_row)
            for j, col in enumerate(cols):
                idx = i + j
                if idx >= len(images):
                    break

                with col:
                    st.image(images[idx], caption=file_names[idx], use_container_width=True)
                    st.info(f"**True: {true_labels[idx]}**")

                    result = results[idx]
                    label = result["label"]
                    confidence = result["confidence"]

                    if label == true_labels[idx]:
                        st.success(f"**Predicted: {label}** ({confidence:.1%})")
                    else:
                        st.error(f"**Predicted: {label}** ({confidence:.1%})")

                    st.plotly_chart(create_stacked_bar(result["probabilities"]), use_container_width=True)

        st.markdown("---")
        st.subheader("Analysis")

        if len(results) > 1:
            col1, col2, col3 = st.columns(3)

            with col1:
                true_counts = {}
                for label in true_labels:
                    true_counts[label] = true_counts.get(label, 0) + 1
                fig = create_pie_chart(true_counts, "True Label Distribution")
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                pred_counts = {}
                for result in results:
                    pred_counts[result["label"]] = pred_counts.get(result["label"], 0) + 1
                fig = create_pie_chart(pred_counts, "Predicted Distribution")
                st.plotly_chart(fig, use_container_width=True)

            with col3:
                correct = sum(1 for r, t in zip(results, true_labels) if r["label"] == t)
                accuracy = correct / len(results)

                st.metric("Overall Accuracy", f"{accuracy:.0%}")
                st.metric("Correct Predictions", f"{correct}/{len(results)}")

                st.markdown("**Per-class Performance:**")
                for tumor_type in ["notumor", "meningioma", "pituitary", "glioma"]:
                    type_total = sum(1 for t in true_labels if t == tumor_type)
                    if type_total > 0:
                        type_correct = sum(1 for r, t in zip(results, true_labels) if t == tumor_type and r["label"] == tumor_type)
                        st.write(f"{tumor_type}: {type_correct}/{type_total}")

        st.markdown("---")
        st.subheader("Training Learning Curves")

        st.markdown("""
        Learning curves show how the model learned during training. They help identify:
        - Model learning progress over epochs
        - Overfitting (training accuracy much higher than validation)
        - Underfitting (both curves plateau at low accuracy)
        - Training stability and optimal duration
        """)

        history = load_training_history()
        if history:
            fig_acc, fig_loss = create_learning_curves(history)

            col1, col2 = st.columns(2)
            with col1:
                st.plotly_chart(fig_acc, use_container_width=True)
            with col2:
                st.plotly_chart(fig_loss, use_container_width=True)

            final_train_acc = history['accuracy'][-1]
            final_val_acc = history['val_accuracy'][-1]
            gap = final_train_acc - final_val_acc

            if gap < 0.05:
                st.success(f"Excellent generalization: Training-validation gap is only {gap*100:.1f}%")
            elif gap < 0.10:
                st.info(f"Good generalization: Training-validation gap is {gap*100:.1f}%")
            else:
                st.warning(f"Some overfitting detected: Training-validation gap is {gap*100:.1f}%")
        else:
            st.info("Training history not available. Run the training script to generate learning curves.")

        st.markdown("---")
        st.subheader("Model Performance Analysis")

        if ANTHROPIC_API_KEY:
            if "messages" not in st.session_state:
                st.session_state.messages = []

            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])

            if prompt := st.chat_input("Ask about model performance, accuracy patterns, or misclassifications..."):
                st.session_state.messages.append({"role": "user", "content": prompt})
                with st.chat_message("user"):
                    st.markdown(prompt)

                with st.chat_message("assistant"):
                    with st.spinner("Analyzing..."):
                        import anthropic

                        client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

                        context = "You are analyzing model performance on a brain tumor classification task. "
                        context += "Help identify patterns, strengths, and weaknesses.\n\n"
                        context += "**Test Results:**\n\n"

                        correct_count = 0
                        for i, (result, true_label) in enumerate(zip(results, true_labels)):
                            pred_label = result['label']
                            is_correct = pred_label == true_label
                            if is_correct:
                                correct_count += 1

                            context += f"Scan {i+1}:\n"
                            context += f"  - True: {true_label.upper()}\n"
                            context += f"  - Predicted: {pred_label.upper()} ({result['confidence']:.1%})\n"
                            context += f"  - Result: {'CORRECT' if is_correct else 'INCORRECT'}\n"

                            if not is_correct:
                                context += "  - Probabilities: "
                                sorted_probs = sorted(result['probabilities'].items(), key=lambda x: -x[1])
                                context += ", ".join([f"{label}: {prob:.1%}" for label, prob in sorted_probs])
                                context += "\n"
                            context += "\n"

                        context += f"\n**Overall Accuracy: {correct_count}/{len(results)} ({correct_count/len(results):.1%})**\n\n"
                        context += "User question: " + prompt

                        response = client.messages.create(
                            model="claude-sonnet-4-20250514",
                            max_tokens=1024,
                            messages=[{"role": "user", "content": context}]
                        )

                        assistant_message = ""
                        for block in response.content:
                            if hasattr(block, 'text'):
                                assistant_message += block.text

                        st.markdown(assistant_message)
                        st.session_state.messages.append({"role": "assistant", "content": assistant_message})
        else:
            st.warning("API key not configured. Set ANTHROPIC_API_KEY in .env file.")

with tab1:
    uploaded_files = st.file_uploader(
        "Upload MRI images (1-10 images)",
        type=["png", "jpg", "jpeg"],
        accept_multiple_files=True,
        key="file_uploader_tab1"
    )

    if uploaded_files:
        if len(uploaded_files) > 10:
            st.warning("Maximum 10 images allowed. Only the first 10 will be processed.")
            uploaded_files = uploaded_files[:10]

        st.markdown(f"**{len(uploaded_files)} {'image' if len(uploaded_files) == 1 else 'images'} uploaded**")

        if st.button("Classify Images", type="primary", key="classify_btn_tab1"):
            with st.spinner("Processing images..."):
                images = []
                file_names = []
                for file in uploaded_files:
                    img = Image.open(file)
                    img_array = np.array(img)
                    images.append(img_array)
                    file_names.append(file.name)

                if len(images) == 1:
                    results = [predict_single(images[0])]
                else:
                    results = predict_batch(images)

                st.session_state['tab1_results'] = results
                st.session_state['tab1_images'] = images
                st.session_state['tab1_file_names'] = file_names
                st.session_state['tab1_report_generated'] = False

        if 'tab1_results' in st.session_state and st.session_state['tab1_results']:
            results = st.session_state['tab1_results']
            images = st.session_state['tab1_images']
            file_names = st.session_state['tab1_file_names']

            st.markdown("---")
            st.subheader("Results")

            cols_per_row = min(len(images), 3)

            for i in range(0, len(images), cols_per_row):
                cols = st.columns(cols_per_row)
                for j, col in enumerate(cols):
                    idx = i + j
                    if idx >= len(images):
                        break

                    with col:
                        st.image(images[idx], caption=file_names[idx], use_container_width=True)

                        result = results[idx]
                        label = result["label"]
                        confidence = result["confidence"]

                        colored_box(label, f"<strong>{label.upper()}</strong><br><br>Confidence: {confidence:.1%}")
                        st.plotly_chart(create_stacked_bar(result["probabilities"]), use_container_width=True)

            if len(results) > 1:
                st.markdown("---")
                st.subheader("Summary")

                pred_counts = {}
                for result in results:
                    pred_counts[result["label"]] = pred_counts.get(result["label"], 0) + 1

                fig = create_pie_chart(pred_counts, "Prediction Distribution")
                st.plotly_chart(fig, use_container_width=True)

            if ANTHROPIC_API_KEY and 'tab1_report_generated' in st.session_state and not st.session_state['tab1_report_generated']:
                with st.spinner("Generating medical report..."):
                    import anthropic
                    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

                    context = "Generate a medical report for these MRI scan results. Use professional formatting with bullet points.\n\n"
                    context += "**Scan Results:**\n\n"
                    for i, result in enumerate(results):
                        context += f"Scan {i+1} ({file_names[i]}): "
                        context += f"{result['label'].title()} with {result['confidence']:.1%} confidence"
                        if result['confidence'] < 0.8:
                            sorted_probs = sorted(result['probabilities'].items(), key=lambda x: -x[1])
                            context += f". Alternative: {sorted_probs[1][0].title()} ({sorted_probs[1][1]:.1%})"
                        context += "\n"

                    context += """
Include these sections with bullet points:

### Patient Scan Summary
- List each scan with findings
- Overall assessment

### Findings
- Detailed findings per scan
- Confidence levels

### Clinical Significance
- Implications for patient care
- Prognosis

### Recommendations
- Next steps
- Specialist referrals
- Follow-up requirements

### Medical Disclaimer
> This analysis is AI-assisted and requires review by a qualified medical specialist before clinical decisions.
"""

                    response = client.messages.create(
                        model="claude-sonnet-4-20250514",
                        max_tokens=2048,
                        messages=[{"role": "user", "content": context}]
                    )

                    report_text = ""
                    for block in response.content:
                        if hasattr(block, 'text'):
                            report_text += block.text

                    st.session_state['tab1_report'] = report_text
                    st.session_state['tab1_report_generated'] = True

                    st.session_state['messages_tab1'] = [
                        {"role": "assistant", "content": report_text}
                    ]

            st.markdown("---")
            st.subheader("Refine Report")
            st.markdown("*Ask Claude to edit the report, add details, or change recommendations*")

            if ANTHROPIC_API_KEY:
                if "messages_tab1" not in st.session_state:
                    st.session_state.messages_tab1 = []

                for message in st.session_state.messages_tab1:
                    with st.chat_message(message["role"]):
                        st.markdown(message["content"])

                if prompt := st.chat_input("Request changes to the report...", key="chat_tab1"):
                    st.session_state.messages_tab1.append({"role": "user", "content": prompt})
                    with st.chat_message("user"):
                        st.markdown(prompt)

                    with st.chat_message("assistant"):
                        with st.spinner("Updating report..."):
                            import anthropic

                            client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

                            conversation_messages = []

                            system_context = "You are editing a medical report. Provide the complete updated report with all sections. Use bullet points.\n\n"

                            for i, msg in enumerate(st.session_state.messages_tab1):
                                if i == 0:
                                    conversation_messages.append({
                                        "role": "user",
                                        "content": system_context + "Initial report:\n\n" + msg["content"]
                                    })
                                    conversation_messages.append({
                                        "role": "assistant",
                                        "content": "Report generated. Request any changes."
                                    })
                                else:
                                    conversation_messages.append(msg)

                            conversation_messages.append({
                                "role": "user",
                                "content": prompt
                            })

                            response = client.messages.create(
                                model="claude-sonnet-4-20250514",
                                max_tokens=2048,
                                messages=conversation_messages
                            )

                            assistant_message = ""
                            for block in response.content:
                                if hasattr(block, 'text'):
                                    assistant_message += block.text

                            st.markdown(assistant_message)
                            st.session_state.messages_tab1.append({"role": "assistant", "content": assistant_message})
                            st.session_state['tab1_report'] = assistant_message
                            st.rerun()
            else:
                st.warning("API key not configured. Set ANTHROPIC_API_KEY in .env file.")

            if 'tab1_report' in st.session_state and st.session_state['tab1_report']:
                st.markdown("---")
                st.subheader("Download Report")

                try:
                    latest_report = st.session_state['tab1_report']
                    if 'messages_tab1' in st.session_state and len(st.session_state['messages_tab1']) > 0:
                        for msg in reversed(st.session_state['messages_tab1']):
                            if msg['role'] == 'assistant':
                                latest_report = msg['content']
                                break

                    with st.spinner("Generating PDF..."):
                        pdf_bytes = generate_pdf_report(latest_report, images, file_names, results)

                        if pdf_bytes and len(pdf_bytes) > 0:
                            st.download_button(
                                label="Download PDF Report",
                                data=pdf_bytes,
                                file_name=f"neurosight_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                                mime="application/pdf",
                                key="download_pdf_tab1",
                                type="primary",
                                use_container_width=True
                            )
                            st.success(f"PDF ready ({len(pdf_bytes) / 1024:.1f} KB)")
                        else:
                            st.error("PDF generation returned empty data.")
                except Exception as e:
                    st.error(f"Error generating PDF: {str(e)}")
                    import traceback
                    st.code(traceback.format_exc())

            st.markdown("---")
            st.subheader("Save Labeled Images to Training Data")
            st.markdown("*Verify the label and save to improve future model training*")

            for idx, (img, file_name, result) in enumerate(zip(images, file_names, results)):
                with st.expander(f"Label Image {idx+1}: {file_name}", expanded=True):
                    col1, col2 = st.columns([1, 2])

                    with col1:
                        st.image(img, use_container_width=True)

                    with col2:
                        st.markdown(f"**Predicted:** {result['label']} ({result['confidence']:.1%})")

                        true_label = st.selectbox(
                            "Select true category:",
                            ["notumor", "meningioma", "pituitary", "glioma"],
                            key=f"label_select_{idx}"
                        )

                        if st.button(f"Save to Training Data", key=f"save_file_{idx}", type="primary"):
                            save_dir = Path(__file__).parent / "data" / "labeled_training" / true_label
                            save_dir.mkdir(parents=True, exist_ok=True)

                            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                            original_name = Path(file_name).stem
                            save_path = save_dir / f"{original_name}_{timestamp}.jpg"

                            img_pil = Image.fromarray(img)
                            if img_pil.mode == 'RGBA':
                                rgb_img = Image.new('RGB', img_pil.size, (255, 255, 255))
                                rgb_img.paste(img_pil, mask=img_pil.split()[3])
                                rgb_img.save(save_path)
                            else:
                                img_pil.convert('RGB').save(save_path)

                            st.success(f"Saved to: {save_path.relative_to(Path(__file__).parent)}")

    else:
        st.info("Please upload one or more MRI brain scan images to get started.")

with tab3:
    st.header("About NeuroSight")

    st.markdown("---")

    st.subheader("Vision")
    st.markdown("""
    Brain tumors require expert interpretation that is time-intensive and prone to variability.
    NeuroSight assists radiologists with fast, accurate classification and provides an integrated
    interface for clinical reasoning and treatment planning.
    """)

    st.markdown("---")

    st.subheader("Problem Statement")
    st.markdown("""
    MRI interpretation is demanding and workloads continue to rise. Subtle tumor patterns are
    easy to overlook, slowing treatment decisions. NeuroSight classifies scans into four
    categories and provides structured support for next steps.
    """)

    st.markdown("**The four categories:**")

    col1, col2 = st.columns(2)

    with col1:
        colored_box('notumor', """
        <strong>No Tumor</strong><br>
        A normal brain scan without abnormal masses.
        """)

        colored_box('meningioma', """
        <strong>Meningioma</strong><br>
        Usually slow-growing tumors from the meninges, treated
        through monitoring or surgery.
        """)

    with col2:
        colored_box('pituitary', """
        <strong>Pituitary Tumor</strong><br>
        Tumors near the pituitary gland that may affect hormone
        regulation.
        """)

        colored_box('glioma', """
        <strong>Glioma</strong><br>
        Tumors from glial cells, often requiring surgery,
        radiotherapy or chemotherapy.
        """)

    st.markdown("---")

    st.subheader("How NeuroSight Works")

    st.markdown("### 1. Upload & Diagnose")
    st.markdown("""
    Upload an MRI scan and the system classifies it with a confidence breakdown.
    A medical assistant interface helps refine reasoning and explore next steps.
    """)

    st.markdown("### 2. Test Model Performance")
    st.markdown("""
    Select random dataset images to test. The system classifies them, compares with
    actuals, and displays accuracy metrics and distributions.
    """)

    st.markdown("### 3. Vision & Explanation")
    st.markdown("""
    This tab explains the project motivation, medical context, and how NeuroSight
    supports radiological workflows.
    """)

st.markdown("---")
st.markdown(
    "<small>Model: Custom CNN trained on Brain Tumor MRI Dataset | For educational purposes only</small>",
    unsafe_allow_html=True
)
