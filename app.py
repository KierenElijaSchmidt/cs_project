import streamlit as st
import numpy as np
from PIL import Image
from pathlib import Path
import random
import os
import time
import io
import re
from dotenv import load_dotenv
import plotly.express as px
import plotly.graph_objects as go
from prediction import predict_single, predict_batch, LABEL_MAP

# Load environment variables
load_dotenv()
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")

# Test data path
TEST_DATA_PATH = Path(__file__).parent / "data" / "brain-tumor-mri-dataset" / "Testing"

def get_random_test_images(n: int) -> list:
    """Get n random images from test set with their true labels."""
    all_images = []
    for class_name in ["glioma", "meningioma", "notumor", "pituitary"]:
        class_dir = TEST_DATA_PATH / class_name
        if class_dir.exists():
            for img_path in class_dir.glob("*.jpg"):
                all_images.append((img_path, class_name))

    selected = random.sample(all_images, min(n, len(all_images)))
    return selected

def create_probability_bar(probabilities: dict, title: str = "Prediction Probabilities"):
    """Create a horizontal stacked bar chart for probabilities."""
    labels = list(probabilities.keys())
    values = list(probabilities.values())

    # Create horizontal bar chart
    fig = go.Figure(go.Bar(
        x=values,
        y=['Prediction'],
        orientation='h',
        text=[f'{v:.1%}' for v in values],
        textposition='inside',
        marker_color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
    ))

    fig.update_layout(
        title=title,
        xaxis_title="Probability",
        height=100,
        margin=dict(l=0, r=0, t=30, b=0),
        showlegend=False,
        xaxis=dict(range=[0, 1], tickformat='.0%')
    )
    return fig

def create_stacked_bar(probabilities: dict):
    """Create a 100% stacked bar chart for probabilities."""
    # Medical-themed color palette
    colors = {'glioma': '#E63946', 'meningioma': '#457B9D', 'notumor': '#2A9D8F', 'pituitary': '#F4A261'}

    fig = go.Figure()
    cumsum = 0
    for label, prob in sorted(probabilities.items(), key=lambda x: -x[1]):
        fig.add_trace(go.Bar(
            name=label,
            x=[prob],
            y=[''],
            orientation='h',
            marker_color=colors.get(label, '#888888'),
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
    """Create a pie chart."""
    # Medical-themed color palette
    colors = {'glioma': '#E63946', 'meningioma': '#457B9D', 'notumor': '#2A9D8F', 'pituitary': '#F4A261'}

    fig = px.pie(
        values=list(data.values()),
        names=list(data.keys()),
        title=title,
        color=list(data.keys()),
        color_discrete_map=colors
    )
    fig.update_traces(textposition='inside', textinfo='percent+label')
    fig.update_layout(height=300, margin=dict(l=20, r=20, t=40, b=20))
    return fig

def get_chat_response(api_key: str, results: list, true_labels: list = None):
    """Get treatment recommendations from Claude."""
    import anthropic

    client = anthropic.Anthropic(api_key=api_key)

    # Build context
    context = "Based on the MRI brain scan analysis:\n\n"
    for i, result in enumerate(results):
        context += f"Image {i+1}: Predicted {result['label']} with {result['confidence']:.1%} confidence"
        if true_labels:
            context += f" (True label: {true_labels[i]})"
        context += "\n"

    return context

st.set_page_config(
    page_title="Brain Tumor Classification",
    page_icon="🧠",
    layout="wide"
)

st.title("🧠 Brain Tumor MRI Classification")
st.markdown("Upload MRI brain scans to classify tumor types using EfficientNetB0.")

# Sidebar info
with st.sidebar:
    st.header("About")
    st.markdown("""
    **Brain Tumor Classification**

    This app uses an EfficientNetB0 model to classify brain MRI scans into:
    - **Glioma** - malignant tumor
    - **Meningioma** - usually benign
    - **Pituitary** - pituitary gland tumor
    - **No Tumor** - healthy scan

    ⚠️ For educational purposes only
    """)

# Tabs for different input methods
tab1, tab2 = st.tabs(["Upload Images", "Random Test Images"])

with tab1:
    # File uploader
    uploaded_files = st.file_uploader(
        "Upload MRI images (1-10 images)",
        type=["png", "jpg", "jpeg"],
        accept_multiple_files=True,
        key="file_uploader_tab1"
    )

with tab2:
    st.markdown("**Select random images from the test dataset**")

    # Slider for number of images
    num_images = st.slider("Number of random images", min_value=1, max_value=10, value=3)

    # Button to load random images
    if st.button("Load Random Test Images", type="primary"):
        test_images = get_random_test_images(num_images)
        st.session_state['test_images'] = test_images

    # Display and process test images if loaded
    if 'test_images' in st.session_state and st.session_state['test_images']:
        test_images = st.session_state['test_images']
        st.markdown(f"**{len(test_images)} image(s) loaded**")

        with st.spinner("Processing images..."):
            # Load images
            images = []
            true_labels = []
            file_names = []
            for img_path, true_label in test_images:
                img = Image.open(img_path)
                img_array = np.array(img)
                images.append(img_array)
                true_labels.append(true_label)
                file_names.append(img_path.name)

            # Get predictions
            if len(images) == 1:
                results = [predict_single(images[0])]
            else:
                results = predict_batch(images)

        # Store results for chat
        st.session_state['results'] = results
        st.session_state['true_labels'] = true_labels

        # Display results
        st.markdown("---")
        st.subheader("Results")

        # Create columns for display
        cols_per_row = min(len(images), 3)

        for i in range(0, len(images), cols_per_row):
            cols = st.columns(cols_per_row)
            for j, col in enumerate(cols):
                idx = i + j
                if idx >= len(images):
                    break

                with col:
                    # Display image
                    st.image(images[idx], caption=file_names[idx], use_container_width=True)

                    # Display true label
                    st.info(f"**True: {true_labels[idx]}**")

                    # Display prediction
                    result = results[idx]
                    label = result["label"]
                    confidence = result["confidence"]

                    # Color code based on correctness
                    if label == true_labels[idx]:
                        st.success(f"**Predicted: {label}** ({confidence:.1%}) ✓")
                    else:
                        st.error(f"**Predicted: {label}** ({confidence:.1%}) ✗")

                    # Probability bar chart
                    st.plotly_chart(create_stacked_bar(result["probabilities"]), use_container_width=True)

        # Analysis section
        st.markdown("---")
        st.subheader("📊 Analysis")

        if len(results) > 1:
            col1, col2, col3 = st.columns(3)

            with col1:
                # True label distribution
                true_counts = {}
                for label in true_labels:
                    true_counts[label] = true_counts.get(label, 0) + 1
                fig = create_pie_chart(true_counts, "True Label Distribution")
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                # Predicted label distribution
                pred_counts = {}
                for result in results:
                    pred_counts[result["label"]] = pred_counts.get(result["label"], 0) + 1
                fig = create_pie_chart(pred_counts, "Predicted Distribution")
                st.plotly_chart(fig, use_container_width=True)

            with col3:
                # Accuracy metrics
                correct = sum(1 for r, t in zip(results, true_labels) if r["label"] == t)
                accuracy = correct / len(results)

                st.metric("Overall Accuracy", f"{accuracy:.0%}")
                st.metric("Correct Predictions", f"{correct}/{len(results)}")

                # Per-class accuracy
                st.markdown("**Per-class Performance:**")
                for tumor_type in ["glioma", "meningioma", "notumor", "pituitary"]:
                    type_total = sum(1 for t in true_labels if t == tumor_type)
                    if type_total > 0:
                        type_correct = sum(1 for r, t in zip(results, true_labels) if t == tumor_type and r["label"] == tumor_type)
                        st.write(f"{tumor_type}: {type_correct}/{type_total}")

        # Chat interface
        st.markdown("---")
        st.subheader("💬 Patient Consultation Assistant")

        if ANTHROPIC_API_KEY:
            # Initialize chat history
            if "messages" not in st.session_state:
                st.session_state.messages = []

            # Display chat history
            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])

            # Chat input
            if prompt := st.chat_input("Ask about treatment options or patient care..."):
                # Add user message
                st.session_state.messages.append({"role": "user", "content": prompt})
                with st.chat_message("user"):
                    st.markdown(prompt)

                # Get AI response
                with st.chat_message("assistant"):
                    with st.spinner("Thinking..."):
                        import anthropic

                        client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

                        # Build context from results - NOTE: Only predictions, no ground truth
                        # This simulates real-world use where the nurse only has AI predictions
                        context = "You are a medical AI assistant helping a nurse with brain tumor diagnosis decisions. "
                        context += "You are assisting with patient care based on MRI classification results. "
                        context += "Based on the AI model's MRI analysis:\n\n"
                        for i, result in enumerate(results):
                            context += f"Scan {i+1}: {result['label'].upper()} (confidence: {result['confidence']:.1%})\n"
                            # Show secondary predictions if confidence is below 80%
                            if result['confidence'] < 0.8:
                                sorted_probs = sorted(result['probabilities'].items(), key=lambda x: -x[1])
                                context += f"  Alternative possibilities: {sorted_probs[1][0]} ({sorted_probs[1][1]:.1%})"
                                if sorted_probs[1][1] > 0.1:
                                    context += f", {sorted_probs[2][0]} ({sorted_probs[2][1]:.1%})"
                                context += "\n"

                        context += "\nProvide helpful guidance on next steps, referrals, and patient care. "
                        context += "Be clear this is AI-assisted analysis and final diagnosis requires specialist review. "
                        context += "Do NOT mention that you have access to 'true labels' or 'ground truth' - respond as if these predictions are the only information available."

                        # Build message content
                        user_message = context + "\n\nUser question: " + prompt

                        response = client.messages.create(
                            model="claude-sonnet-4-20250514",
                            max_tokens=1024,
                            messages=[
                                {"role": "user", "content": user_message}
                            ]
                        )

                        # Get the text content from the response
                        assistant_message = ""
                        for block in response.content:
                            if hasattr(block, 'text'):
                                assistant_message += block.text

                        st.markdown(assistant_message)
                        st.session_state.messages.append({"role": "assistant", "content": assistant_message})
        else:
            st.warning("API key not configured. Set ANTHROPIC_API_KEY in .env file.")

with tab1:
    if uploaded_files:
        # Limit to 10 images
        if len(uploaded_files) > 10:
            st.warning("Maximum 10 images allowed. Only the first 10 will be processed.")
            uploaded_files = uploaded_files[:10]

        st.markdown(f"**{len(uploaded_files)} image(s) uploaded**")

        # Process button
        if st.button("Classify Images", type="primary", key="classify_btn_tab1"):
            with st.spinner("Processing images..."):
                # Load images
                images = []
                file_names = []
                for file in uploaded_files:
                    img = Image.open(file)
                    img_array = np.array(img)
                    images.append(img_array)
                    file_names.append(file.name)

                # Get predictions
                if len(images) == 1:
                    results = [predict_single(images[0])]
                else:
                    results = predict_batch(images)

                # Store in session state
                st.session_state['tab1_results'] = results
                st.session_state['tab1_images'] = images
                st.session_state['tab1_file_names'] = file_names
                st.session_state['tab1_report_generated'] = False  # Flag to trigger report generation

        # Display results from session state
        if 'tab1_results' in st.session_state and st.session_state['tab1_results']:
            results = st.session_state['tab1_results']
            images = st.session_state['tab1_images']
            file_names = st.session_state['tab1_file_names']

            st.markdown("---")
            st.subheader("Results")

            # Create columns for display
            cols_per_row = min(len(images), 3)

            for i in range(0, len(images), cols_per_row):
                cols = st.columns(cols_per_row)
                for j, col in enumerate(cols):
                    idx = i + j
                    if idx >= len(images):
                        break

                    with col:
                        # Display image
                        st.image(images[idx], caption=file_names[idx], use_container_width=True)

                        # Display prediction
                        result = results[idx]
                        label = result["label"]
                        confidence = result["confidence"]

                        # Neutral design with gradient background
                        st.markdown(f"""
                        <div style="
                            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                            padding: 15px;
                            border-radius: 10px;
                            text-align: center;
                            color: white;
                            margin: 10px 0;
                        ">
                            <h3 style="margin: 0; font-size: 1.2em;">{label.upper()}</h3>
                            <p style="margin: 5px 0 0 0; font-size: 0.9em;">Confidence: {confidence:.1%}</p>
                        </div>
                        """, unsafe_allow_html=True)

                        # Probability bar chart
                        st.plotly_chart(create_stacked_bar(result["probabilities"]), use_container_width=True)

            # Summary statistics
            if len(results) > 1:
                st.markdown("---")
                st.subheader("Summary")

                # Pie chart of predictions
                pred_counts = {}
                for result in results:
                    pred_counts[result["label"]] = pred_counts.get(result["label"], 0) + 1

                fig = create_pie_chart(pred_counts, "Prediction Distribution")
                st.plotly_chart(fig, use_container_width=True)

            # Auto-generate report after displaying predictions (if not already generated)
            if ANTHROPIC_API_KEY and 'tab1_report_generated' in st.session_state and not st.session_state['tab1_report_generated']:
                with st.spinner("Generating medical report..."):
                    import anthropic
                    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

                    # Build context for report generation with improved formatting
                    context = """You are a medical AI assistant. Generate a comprehensive, well-formatted medical report based on the following MRI brain scan analysis.

IMPORTANT FORMATTING GUIDELINES:
- Use proper title case for section headers (not ALL CAPS)
- Use markdown formatting with ### for main sections, #### for subsections
- Add horizontal lines (---) between major sections for visual separation
- Use bullet points (•) for lists and findings
- Add subtle box elements using > blockquotes for important notes
- Include relevant medical icons/symbols where appropriate
- Keep formatting clean, professional, and easy to read
- Prioritize medical functionality while maintaining aesthetic appeal

"""
                    context += "**MRI Analysis Results:**\n\n"
                    for i, result in enumerate(results):
                        context += f"**Scan {i+1}** ({file_names[i]}):\n"
                        context += f"  • Primary diagnosis: **{result['label'].title()}**\n"
                        context += f"  • Confidence level: {result['confidence']:.1%}\n"
                        if result['confidence'] < 0.8:
                            sorted_probs = sorted(result['probabilities'].items(), key=lambda x: -x[1])
                            context += f"  • Alternative possibilities: {sorted_probs[1][0].title()} ({sorted_probs[1][1]:.1%})"
                            if len(sorted_probs) > 2 and sorted_probs[1][1] > 0.1:
                                context += f", {sorted_probs[2][0].title()} ({sorted_probs[2][1]:.1%})"
                            context += "\n"
                        context += "\n"

                    context += """\nGenerate a professional medical report with these sections:

### Patient Scan Summary
Brief overview of all scans analyzed

### Findings
Detailed findings for each scan with clinical observations

### Clinical Significance
What these findings mean for patient care and prognosis

### Recommendations
Next steps, specialist referrals, and follow-up care needed

### Medical Disclaimer
> Note that this is AI-assisted analysis requiring specialist review

Format the report beautifully while maintaining medical professionalism."""

                    response = client.messages.create(
                        model="claude-sonnet-4-20250514",
                        max_tokens=2048,
                        messages=[
                            {"role": "user", "content": context}
                        ]
                    )

                    # Extract report text
                    report_text = ""
                    for block in response.content:
                        if hasattr(block, 'text'):
                            report_text += block.text

                    # Store report in session state
                    st.session_state['tab1_report'] = report_text
                    st.session_state['tab1_report_generated'] = True

                    # Initialize chat history with the report
                    st.session_state['messages_tab1'] = [
                        {"role": "assistant", "content": report_text}
                    ]

            # Medical Report Section
            st.markdown("---")
            st.subheader("📋 Medical Report")

            if 'tab1_report' in st.session_state and st.session_state['tab1_report']:
                # Add PDF download button
                col1, col2 = st.columns([4, 1])
                with col2:
                    if st.button("📄 Download PDF", key="download_pdf_btn", type="secondary"):
                        # Generate PDF with report and images
                        try:
                            from fpdf import FPDF

                            class PDF(FPDF):
                                def header(self):
                                    self.set_font('Arial', 'B', 16)
                                    self.cell(0, 10, 'Medical MRI Report', 0, 1, 'C')
                                    self.ln(5)

                                def footer(self):
                                    self.set_y(-15)
                                    self.set_font('Arial', 'I', 8)
                                    self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')

                            pdf = FPDF()
                            pdf.add_page()
                            pdf.set_font("Arial", size=11)

                            # Add report content (strip markdown for PDF)
                            report_text = st.session_state['tab1_report']
                            # Simple markdown to plain text conversion
                            report_text = re.sub(r'#{1,6}\s', '', report_text)  # Remove headers
                            report_text = re.sub(r'\*\*(.+?)\*\*', r'\1', report_text)  # Remove bold
                            report_text = re.sub(r'\*(.+?)\*', r'\1', report_text)  # Remove italics
                            report_text = re.sub(r'>\s', '', report_text)  # Remove blockquotes

                            # Add text with line breaks
                            pdf.multi_cell(0, 5, report_text.encode('latin-1', 'replace').decode('latin-1'))

                            pdf.ln(10)
                            pdf.set_font("Arial", 'B', 12)
                            pdf.cell(0, 10, 'Image Classifications:', 0, 1)
                            pdf.set_font("Arial", size=10)

                            # Add images and their classifications
                            for idx, (img, file_name, result) in enumerate(zip(images, file_names, results)):
                                pdf.ln(5)
                                pdf.cell(0, 5, f"Scan {idx+1}: {file_name}", 0, 1)
                                pdf.cell(0, 5, f"Classification: {result['label'].title()} ({result['confidence']:.1%})", 0, 1)

                                # Save image temporarily and add to PDF
                                temp_img_path = f"/tmp/temp_img_{idx}.jpg"
                                img_pil = Image.fromarray(img)
                                if img_pil.mode == 'RGBA':
                                    rgb_img = Image.new('RGB', img_pil.size, (255, 255, 255))
                                    rgb_img.paste(img_pil, mask=img_pil.split()[3])
                                    rgb_img.save(temp_img_path)
                                else:
                                    img_pil.convert('RGB').save(temp_img_path)

                                # Add image to PDF (scaled to fit)
                                pdf.image(temp_img_path, x=10, w=100)
                                pdf.ln(5)

                            # Save PDF to bytes
                            pdf_output = pdf.output(dest='S').encode('latin-1')

                            st.download_button(
                                label="💾 Click to Download PDF",
                                data=pdf_output,
                                file_name=f"medical_report_{int(time.time())}.pdf",
                                mime="application/pdf",
                                key="final_pdf_download"
                            )
                        except ImportError:
                            st.error("PDF generation requires 'fpdf' library. Install with: pip install fpdf")
                        except Exception as e:
                            st.error(f"Error generating PDF: {str(e)}")
            else:
                st.info("Report will be generated automatically after classification.")

            # Chat interface for report editing
            st.markdown("---")
            st.subheader("💬 Refine Report with AI Assistant")
            st.markdown("*You can ask Claude to edit the report, add details, change recommendations, etc.*")

            if ANTHROPIC_API_KEY:
                # Initialize chat history for tab1 if not exists
                if "messages_tab1" not in st.session_state:
                    st.session_state.messages_tab1 = []

                # Display chat history (including the initial report)
                for message in st.session_state.messages_tab1:
                    with st.chat_message(message["role"]):
                        st.markdown(message["content"])

                # Chat input
                if prompt := st.chat_input("Request changes to the report (e.g., 'Add more detail to the recommendations section')...", key="chat_tab1"):
                    # Add user message
                    st.session_state.messages_tab1.append({"role": "user", "content": prompt})
                    with st.chat_message("user"):
                        st.markdown(prompt)

                    # Get AI response
                    with st.chat_message("assistant"):
                        with st.spinner("Updating report..."):
                            import anthropic

                            client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

                            # Build conversation history with context
                            conversation_messages = []

                            # Add system context as first user message
                            system_context = "You are a medical AI assistant helping refine a medical report. "
                            system_context += "The user can request edits, additions, or clarifications to the report. "
                            system_context += "When editing, provide the COMPLETE updated report with all sections, not just the changed parts. "
                            system_context += "Maintain professional medical report formatting.\n\n"

                            # Add all previous messages
                            for i, msg in enumerate(st.session_state.messages_tab1):
                                if i == 0:
                                    # First message is the original report
                                    conversation_messages.append({
                                        "role": "user",
                                        "content": system_context + "Here is the initial report:\n\n" + msg["content"]
                                    })
                                    conversation_messages.append({
                                        "role": "assistant",
                                        "content": "I've generated the medical report. You can ask me to make any changes or additions you'd like."
                                    })
                                else:
                                    conversation_messages.append(msg)

                            # Add current user prompt
                            conversation_messages.append({
                                "role": "user",
                                "content": prompt
                            })

                            response = client.messages.create(
                                model="claude-sonnet-4-20250514",
                                max_tokens=2048,
                                messages=conversation_messages
                            )

                            # Get the text content from the response
                            assistant_message = ""
                            for block in response.content:
                                if hasattr(block, 'text'):
                                    assistant_message += block.text

                            st.markdown(assistant_message)
                            st.session_state.messages_tab1.append({"role": "assistant", "content": assistant_message})

                            # Update the stored report with the latest version
                            st.session_state['tab1_report'] = assistant_message
            else:
                st.warning("API key not configured. Set ANTHROPIC_API_KEY in .env file.")

            # Download labeled images section
            st.markdown("---")
            st.subheader("📥 Download Labeled Images")
            st.markdown("*Once you're satisfied with the report, label and download the images for training data.*")

            # Create columns for labeling each image
            for idx, (img, file_name, result) in enumerate(zip(images, file_names, results)):
                with st.expander(f"Label Image {idx+1}: {file_name}", expanded=True):
                    col1, col2 = st.columns([1, 2])

                    with col1:
                        st.image(img, use_container_width=True)

                    with col2:
                        st.markdown(f"**Predicted:** {result['label']} ({result['confidence']:.1%})")

                        true_label = st.selectbox(
                            "Select true category:",
                            ["glioma", "meningioma", "notumor", "pituitary"],
                            key=f"label_select_{idx}"
                        )

                        # Convert image for download
                        img_pil = Image.fromarray(img)
                        if img_pil.mode == 'RGBA':
                            rgb_img = Image.new('RGB', img_pil.size, (255, 255, 255))
                            rgb_img.paste(img_pil, mask=img_pil.split()[3])
                            img_pil = rgb_img
                        else:
                            img_pil = img_pil.convert('RGB')

                        # Save to bytes
                        buf = io.BytesIO()
                        img_pil.save(buf, format='JPEG')
                        buf.seek(0)

                        # Direct download button (no nested button)
                        st.download_button(
                            label=f"📥 Download as {true_label}.jpg",
                            data=buf.getvalue(),
                            file_name=f"{true_label}.jpg",
                            mime="image/jpeg",
                            key=f"download_file_{idx}",
                            type="primary"
                        )

    else:
        st.info("Please upload one or more MRI brain scan images to get started.")

# Footer
st.markdown("---")
st.markdown(
    "<small>Model: EfficientNetB0 trained on Brain Tumor MRI Dataset | ⚠️ For educational purposes only</small>",
    unsafe_allow_html=True
)
