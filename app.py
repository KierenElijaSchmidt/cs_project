import streamlit as st
import numpy as np
from PIL import Image
from pathlib import Path
import random
import os
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
        accept_multiple_files=True
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

                        # Build messages
                        messages = [{"role": "user", "content": context + "\n\nUser question: " + prompt}]

                        response = client.messages.create(
                            model="claude-sonnet-4-20250514",
                            max_tokens=1024,
                            messages=messages
                        )

                        assistant_message = response.content[0].text
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
        if st.button("Classify Images", type="primary"):
            with st.spinner("Processing images..."):
                # Load images
                images = []
                for file in uploaded_files:
                    img = Image.open(file)
                    img_array = np.array(img)
                    images.append(img_array)

                # Get predictions
                if len(images) == 1:
                    results = [predict_single(images[0])]
                else:
                    results = predict_batch(images)

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
                        st.image(images[idx], caption=uploaded_files[idx].name, use_container_width=True)

                        # Display prediction
                        result = results[idx]
                        label = result["label"]
                        confidence = result["confidence"]

                        # Color code based on tumor presence
                        if label == "notumor":
                            st.success(f"**{label}** ({confidence:.1%})")
                        else:
                            st.error(f"**{label}** ({confidence:.1%})")

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

    else:
        st.info("Please upload one or more MRI brain scan images to get started.")

# Footer
st.markdown("---")
st.markdown(
    "<small>Model: EfficientNetB0 trained on Brain Tumor MRI Dataset | ⚠️ For educational purposes only</small>",
    unsafe_allow_html=True
)
