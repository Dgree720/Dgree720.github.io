# pip install numpy pandas gensim
# streamlit run 17_2LLM02.py
import streamlit as st
import numpy as np
import plotly.graph_objects as go
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import pandas as pd
import gensim.downloader as api

# Page configuration
st.set_page_config(
    page_title="Word Vector 3D Visualization", layout="wide", page_icon="📊"
)
st.title("📊 Word Vector 3D Visualization|Student ID|")
st.markdown("Enter words to automatically vectorize and visualize them in 3D space")


# Cache Word2Vec model loading
@st.cache_resource
def load_word2vec_model(model_name):
    """Load Word2Vec pre-trained model"""
    with st.spinner(
        f"Loading {model_name} model... (First time download may take a few minutes)"
    ):
        model = api.load(model_name)
    return model


# Sidebar settings
with st.sidebar:
    st.header("⚙️ Settings")

    # Vectorization method selection
    method = st.selectbox(
        "Vectorization Method",
        [
            "Word2Vec (glove-wiki-gigaword-50)",
            "Word2Vec (word2vec-google-news-300)",
            "Word2Vec (glove-twitter-25)",
            "Character Features",
            "Letter Frequency",
            "N-gram",
        ],
        help="Choose the method to convert words into vectors",
    )

    st.divider()
    # Default words
    if "Word2Vec" in method:
        default_words = """king
queen
man
woman
prince
princess
boy
girl
father
mother
son
daughter
brother
sister
husband
wife"""
    else:
        default_words = """king
queen
man
woman
boy
girl
prince
princess
father
mother
brother
sister"""
    words_input = st.text_area(
        "Enter Words (one per line)",
        value=default_words,
        height=300,
        help="Enter one English word per line",
    )

    st.divider()
    # 3D chart settings
    st.subheader("Visualization Settings")
    marker_size = st.slider("Marker Size", 5, 20, 10)
    show_labels = st.checkbox("Show Labels", value=True)

    if "Word2Vec" in method:
        st.divider()
        show_similar = st.checkbox("Show Most Similar Words", value=False)
        if show_similar:
            similarity_count = st.slider("Number of Similar Words", 1, 10, 5)


# Word vectorization functions
def vectorize_word_features(word):
    """Vectorization based on character features"""
    features = []
    word_lower = word.lower()

    # Basic features
    features.append(len(word))
    features.append(len(set(word_lower)))
    # Vowels and consonants
    vowels = sum(1 for c in word_lower if c in "aeiou")
    consonants = sum(1 for c in word_lower if c.isalpha() and c not in "aeiou")
    features.append(vowels)
    features.append(consonants)

    # Letter position features
    if word_lower:
        features.append(ord(word_lower[0]) - ord("a"))
        features.append(ord(word_lower[-1]) - ord("a"))
    else:
        features.extend([0, 0])

    # ASCII sum (normalized)
    features.append(sum(ord(c) for c in word_lower) / 100)

    # Specific letter presence
    common_letters = "etaoinshrdlu"
    for letter in common_letters:
        features.append(1 if letter in word_lower else 0)
    return features


def vectorize_word_frequency(word):
    """Vectorization based on letter frequency"""
    word_lower = word.lower()
    vector = []

    for letter in "abcdefghijklmnopqrstuvwxyz":
        vector.append(word_lower.count(letter))
    return vector


def vectorize_word_ngram(word):
    """Vectorization based on N-grams"""
    word_lower = word.lower()
    features = []

    common_bigrams = ["th", "he", "in", "er", "an", "re", "on", "at", "en", "nd"]
    for bigram in common_bigrams:
        features.append(word_lower.count(bigram))

    common_trigrams = ["the", "ing", "and", "ion", "tio", "ent", "her"]
    for trigram in common_trigrams:
        features.append(word_lower.count(trigram))

    features.append(len(word))
    features.append(len(set(word_lower)))
    return features


# Process input words
words = [w.strip() for w in words_input.split("\n") if w.strip()]

if len(words) < 3:
    st.warning("⚠️ Please enter at least 3 words")
    st.stop()

# Vectorization based on selected method
if "Word2Vec" in method:
    # Select model
    if "glove-wiki-gigaword-50" in method:
        model_name = "glove-wiki-gigaword-50"
    elif "word2vec-google-news-300" in method:
        model_name = "word2vec-google-news-300"
    else:
        model_name = "glove-twitter-25"

    # Load model
    try:
        w2v_model = load_word2vec_model(model_name)

        # Check which words are in the model
        valid_words = []
        missing_words = []
        vectors_list = []

        for word in words:
            word_lower = word.lower()
            if word_lower in w2v_model:
                valid_words.append(word)
                vectors_list.append(w2v_model[word_lower])
            else:
                missing_words.append(word)

        if len(valid_words) < 3:
            st.error(
                f"❌ Too few words found in the model ({len(valid_words)} words). Please enter more common English words."
            )
            if missing_words:
                st.warning(f"⚠️ Words not found in model: {', '.join(missing_words)}")
            st.stop()

        if missing_words:
            st.info(f"ℹ️ Words not found in model (ignored): {', '.join(missing_words)}")

        words = valid_words
        vectors = np.array(vectors_list)

        # Show similar words
        if show_similar:
            st.subheader("🔍 Most Similar Words")
            for word in words[:3]:  # Show similar words for first 3 words only
                try:
                    similar_words = w2v_model.most_similar(
                        word.lower(), topn=similarity_count
                    )
                    with st.expander(f"Words most similar to '{word}'"):
                        similar_df = pd.DataFrame(
                            similar_words, columns=["Word", "Similarity"]
                        )
                        st.dataframe(
                            similar_df.style.format({"Similarity": "{:.4f}"}),
                            width="stretch",
                        )
                except:
                    pass

    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        st.stop()

else:
    # Select vectorization method
    if method == "Character Features":
        vectorize_func = vectorize_word_features
    elif method == "Letter Frequency":
        vectorize_func = vectorize_word_frequency
    else:
        vectorize_func = vectorize_word_ngram

    # Vectorize all words
    with st.spinner("Vectorizing words..."):
        vectors = np.array([vectorize_func(word) for word in words])

# Standardization and dimensionality reduction
with st.spinner("Performing PCA dimensionality reduction..."):
    # Standardization
    scaler = StandardScaler()
    vectors_scaled = scaler.fit_transform(vectors)

    # PCA dimensionality reduction to 3D
    pca = PCA(n_components=3)
    vectors_3d = pca.fit_transform(vectors_scaled)

# Create DataFrame
df = pd.DataFrame(
    {"Word": words, "X": vectors_3d[:, 0], "Y": vectors_3d[:, 1], "Z": vectors_3d[:, 2]}
)

# Display metrics
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Number of Words", len(words))
with col2:
    st.metric("Original Dimensions", vectors.shape[1])
with col3:
    st.metric("Reduced Dimensions", 3)
with col4:
    variance_explained = sum(pca.explained_variance_ratio_) * 100
    st.metric("Explained Variance", f"{variance_explained:.1f}%")

# Create 3D scatter plot
fig = go.Figure()

# Add scatter points
fig.add_trace(
    go.Scatter3d(
        x=df["X"],
        y=df["Y"],
        z=df["Z"],
        mode="markers+text" if show_labels else "markers",
        marker=dict(
            size=marker_size,
            color=df["Z"],
            colorscale="Viridis",
            showscale=True,
            colorbar=dict(title="Z Axis"),
            line=dict(color="white", width=2),
        ),
        text=df["Word"] if show_labels else None,
        textposition="top center",
        textfont=dict(size=10),
        hovertemplate="<b>%{text}</b><br>"
        + "X: %{x:.3f}<br>"
        + "Y: %{y:.3f}<br>"
        + "Z: %{z:.3f}<br>"
        + "<extra></extra>",
        name="Words",
    )
)

# Update layout
fig.update_layout(
    title=dict(text="Word Vectors in 3D Space", x=0.5, xanchor="center"),
    scene=dict(
        xaxis=dict(title="PC1", backgroundcolor="rgb(230, 230,230)", gridcolor="white"),
        yaxis=dict(title="PC2", backgroundcolor="rgb(230, 230,230)", gridcolor="white"),
        zaxis=dict(title="PC3", backgroundcolor="rgb(230, 230,230)", gridcolor="white"),
        camera=dict(eye=dict(x=1.5, y=1.5, z=1.5)),
    ),
    height=700,
    hovermode="closest",
    showlegend=False,
)

# Display chart
st.plotly_chart(fig, width="stretch")

# Word2Vec special feature: calculate similarity
if "Word2Vec" in method and len(words) >= 2:
    with st.expander("🎯 Calculate Word Similarity"):
        col1, col2 = st.columns(2)
        with col1:
            word1 = st.selectbox("Select first word", words, key="word1")
        with col2:
            word2 = st.selectbox("Select second word", words, key="word2")

        if word1 != word2:
            try:
                similarity = w2v_model.similarity(word1.lower(), word2.lower())
                st.metric(
                    f"Similarity between '{word1}' and '{word2}'",
                    f"{similarity:.4f}",
                    help="Similarity ranges from -1 to 1, closer to 1 means more similar",
                )

                # Visualize similarity
                fig_sim = go.Figure(
                    go.Indicator(
                        mode="gauge+number",
                        value=similarity,
                        domain={"x": [0, 1], "y": [0, 1]},
                        gauge={
                            "axis": {"range": [-1, 1]},
                            "bar": {"color": "darkblue"},
                            "steps": [
                                {"range": [-1, 0], "color": "lightgray"},
                                {"range": [0, 0.5], "color": "lightyellow"},
                                {"range": [0.5, 1], "color": "lightgreen"},
                            ],
                            "threshold": {
                                "line": {"color": "red", "width": 4},
                                "thickness": 0.75,
                                "value": 0.8,
                            },
                        },
                    )
                )
                fig_sim.update_layout(height=300)
                st.plotly_chart(fig_sim, width="stretch")
            except:
                st.error("Unable to calculate similarity")

# Display PCA information
with st.expander("📈 View PCA Principal Component Information"):
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Explained Variance Ratio")
        variance_df = pd.DataFrame(
            {
                "Principal Component": [f"PC{i + 1}" for i in range(3)],
                "Explained Variance Ratio": pca.explained_variance_ratio_,
                "Cumulative Explained Variance": np.cumsum(
                    pca.explained_variance_ratio_
                ),
            }
        )
        st.dataframe(
            variance_df.style.format(
                {
                    "Explained Variance Ratio": "{:.2%}",
                    "Cumulative Explained Variance": "{:.2%}",
                }
            ),
            width="stretch",
        )

    with col2:
        st.subheader("Coordinate Data")
        st.dataframe(
            df.style.format({"X": "{:.4f}", "Y": "{:.4f}", "Z": "{:.4f}"}),
            width="stretch",
            height=300,
        )

# Download data
st.divider()
col1, col2 = st.columns(2)

with col1:
    csv = df.to_csv(index=False)
    st.download_button(
        label="📥 Download 3D Coordinates (CSV)",
        data=csv,
        file_name="word_vectors_3d.csv",
        mime="text/csv",
    )

with col2:
    # Raw vector data
    vectors_df = pd.DataFrame(vectors, index=words)
    vectors_csv = vectors_df.to_csv()
    st.download_button(
        label="📥 Download Raw Vectors (CSV)",
        data=vectors_csv,
        file_name="word_vectors_raw.csv",
        mime="text/csv",
    )

# Usage instructions
with st.expander("ℹ️ How to Use"):
    st.markdown("""
    ### Instructions

    1. **Select Vectorization Method**:
       - **Word2Vec Models**: Use pre-trained word embedding models (Recommended)
         - `glove-wiki-gigaword-50`: 50 dimensions, Wikipedia trained (Fast)
         - `word2vec-google-news-300`: 300 dimensions, Google News trained (Most accurate)
         - `glove-twitter-25`: 25 dimensions, Twitter trained (Lightweight)
       - **Character Features/Letter Frequency/N-gram**: Simple feature engineering methods

    2. **Enter Words**: Enter one English word per line in the left text box

    3. **Adjust Settings**: Adjust marker size and label display options

    4. **Interactive Operations**:
       - Drag to rotate the 3D chart
       - Mouse wheel to zoom
       - Hover to view detailed information

    ### Word2Vec Special Features

    - **Similar Word Search**: View other words most similar to your input words
    - **Similarity Calculation**: Calculate similarity between any two words (cosine similarity)
    - **Semantic Relationships**: Similar words will be closer together in 3D space

    ### Try These Word Combinations

    1. **Gender Relations**: king, queen, man, woman, prince, princess
    2. **Family Relations**: father, mother, son, daughter, brother, sister
    3. **Animals**: dog, cat, lion, tiger, elephant, mouse
    4. **Colors**: red, blue, green, yellow, orange, purple
    5. **Countries & Capitals**: france, paris, germany, berlin, japan, tokyo

    ### Technical Details

    - Uses **PCA (Principal Component Analysis)** for dimensionality reduction to 3D
    - **Standardization** ensures consistent feature scales
    - **Explained Variance** shows information retained after reduction
    - First-time model download may take several minutes
    - Models are cached for faster loading afterwards
    """)

# Footer
st.divider()
st.markdown(
    "<p style='text-align: center; color: gray;'>Word Vector 3D Visualization Tool | Built with Streamlit & Gensim</p>",
    unsafe_allow_html=True,
)

# Required packages section
st.divider()
with st.expander("📦 Required Packages"):
    st.markdown("""
    ### Installation

    Install all required packages using pip:

    ```bash
    pip install streamlit numpy plotly scikit-learn pandas gensim
    ```

    ### Package List

    - **streamlit**: Web application framework
    - **numpy**: Numerical computing library
    - **plotly**: Interactive visualization library
    - **scikit-learn**: Machine learning library (PCA, StandardScaler)
    - **pandas**: Data manipulation library
    - **gensim**: NLP library for Word2Vec models

    ### Version Information (Recommended)

    ```bash
    streamlit>=1.28.0
    numpy>=1.24.0
    plotly>=5.17.0
    scikit-learn>=1.3.0
    pandas>=2.0.0
    gensim>=4.3.0
    ```

    ### How to Run
    
    ```bash
    streamlit run 17_1LLM02.py
    ```
    """)
