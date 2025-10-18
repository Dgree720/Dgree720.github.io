# pip install plotly scikit-learn
# streamlit run 17_1LLM01.py
import streamlit as st
import plotly.graph_objects as go
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import re

st.title("📊 Word Vectorization & 3D Visualization")
st.write("Visualize individual word embeddings in 3D space")

# Sidebar for configuration
st.sidebar.header("Settings")

# Default sample text
default_text = """king queen man woman machine learning prince deep neural network 
princess data python husband wife"""
# Text input
st.sidebar.subheader("Input Text")
text_input = st.sidebar.text_area(
    "Enter text (10+ words recommended):", height=200, value=default_text
)

# Parameters
num_words = st.sidebar.slider(
    "Number of words to visualize", min_value=3, max_value=50, value=10
)
method = st.sidebar.selectbox("Dimensionality reduction method", ["t-SNE", "PCA"])
random_state = st.sidebar.number_input("Random Seed", value=42, min_value=0)


def extract_words(text, n_words=10):
    """Extract individual words from text"""
    # Remove extra whitespace and split into words
    words = re.findall(r"\b\w+\b", text.lower())
    # Remove duplicates while preserving order
    seen = set()
    unique_words = []
    for word in words:
        if word not in seen and len(word) > 1:  # Filter out single-letter words
            seen.add(word)
            unique_words.append(word)

    return unique_words[:n_words]


# Main content
if not text_input.strip():
    st.error("Please enter some text!")
else:
    # Extract words
    words = extract_words(text_input, num_words)

    if len(words) < 3:
        st.error(
            f"Not enough words! Please enter at least 3 unique words (found {len(words)})."
        )
    else:
        st.success(f"✅ Extracted {len(words)} unique words")

        # Show words
        with st.expander("📝 View all words"):
            st.write(", ".join(words))

        if st.button("🚀 Generate 3D Visualization", type="primary"):
            with st.spinner("Processing..."):
                # Step 1: Create context for each word (using surrounding context)
                st.write("### Step 1: Word Vectorization (TF-IDF)")
                st.info("Creating context vectors for each word...")

                # Create simple context by treating each word as a document
                # For better results, you could use word2vec or similar
                vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(2, 4))

                try:
                    word_vectors = vectorizer.fit_transform(words)
                    st.info(f"Original vector dimensions: {word_vectors.shape}")
                except:
                    st.error("Error in vectorization. Try adding more diverse words.")
                    st.stop()

                # Step 2: Dimensionality Reduction
                st.write(f"### Step 2: Dimensionality Reduction ({method})")

                # Adjust perplexity for t-SNE
                if method == "t-SNE":
                    perplexity = min(5, len(words) - 1)
                    if perplexity < 2:
                        perplexity = 2
                    reducer = TSNE(
                        n_components=3, random_state=random_state, perplexity=perplexity
                    )
                else:
                    reducer = PCA(n_components=3, random_state=random_state)

                vectors_3d = reducer.fit_transform(word_vectors.toarray())
                st.info(f"Reduced dimensions: {vectors_3d.shape}")

                # Step 3: 3D Visualization with Plotly
                st.write("### Step 3: Interactive 3D Visualization")
                st.write("💡 Use mouse to rotate, zoom, and pan the 3D plot")

                # Create 3D scatter plot
                fig = go.Figure(
                    data=[
                        go.Scatter3d(
                            x=vectors_3d[:, 0],
                            y=vectors_3d[:, 1],
                            z=vectors_3d[:, 2],
                            mode="markers+text",
                            marker=dict(
                                size=12,
                                color=list(range(len(words))),
                                colorscale="Viridis",
                                showscale=True,
                                colorbar=dict(title="Word Index"),
                                line=dict(color="white", width=1),
                            ),
                            text=words,
                            textposition="top center",
                            textfont=dict(size=12, color="black", family="Arial Black"),
                            hovertext=[
                                f"Word {i}: {word}" for i, word in enumerate(words)
                            ],
                            hoverinfo="text",
                        )
                    ]
                )

                fig.update_layout(
                    title=f"Word Vectors in 3D Space (Each point = 1 word)",
                    scene=dict(
                        xaxis_title="Dimension 1",
                        yaxis_title="Dimension 2",
                        zaxis_title="Dimension 3",
                        camera=dict(eye=dict(x=1.5, y=1.5, z=1.5)),
                    ),
                    width=900,
                    height=700,
                    hovermode="closest",
                )

                st.plotly_chart(fig, use_container_width=True)

                # Show statistics
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Total Words Visualized", len(words))
                with col2:
                    st.metric("Vector Dimensions", "3D")

                # Show coordinates
                with st.expander("📊 View 3D Coordinates"):
                    import pandas as pd

                    df = pd.DataFrame(vectors_3d, columns=["X", "Y", "Z"])
                    df.insert(0, "Word", words)
                    st.dataframe(df, use_container_width=True)

                # Distance matrix
                with st.expander("📏 Word Similarity (Euclidean Distance)"):
                    from scipy.spatial.distance import cdist

                    distances = cdist(vectors_3d, vectors_3d, metric="euclidean")
                    dist_df = pd.DataFrame(distances, columns=words, index=words)
                    st.dataframe(
                        dist_df.style.background_gradient(cmap="YlOrRd_r"),
                        use_container_width=True,
                    )

# Instructions
st.sidebar.markdown("---")
st.sidebar.markdown("""
### 📖 How to use:
1. Enter text with multiple words
2. Adjust number of words to show
3. Choose reduction method:
   - **t-SNE**: Better for clusters
   - **PCA**: Faster, linear
4. Click "Generate 3D Visualization"
5. Mouse controls:
   - **Rotate**: Click and drag
   - **Zoom**: Scroll wheel
   - **Pan**: Right-click and drag
   - **Hover**: See word info

### 💡 Tips:
- Similar words will appear closer
- Each point = one word
- Try related words to see clusters
""")

# Footer
st.markdown("---")
st.markdown("Built with Streamlit + Plotly + scikit-learn")
