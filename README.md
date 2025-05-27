
---

# LangChain AI Agent with Vector Search & Visualization

This project implements a LangChain-based intelligent agent capable of performing chat-based question answering and semantic search on a given document dataset. It leverages vector embeddings, a Chroma vector store, and explores two different similarity search techniques: Locality Sensitive Hashing (LSH) using FAISS and Cosine Similarity using scikit-learn. Search results are visually represented using t-SNE dimensionality reduction plots.

## 🌟 Features

*   **Document Processing:** Efficient loading and chunking of text documents from a local knowledge base.
*   **Hugging Face Embeddings:** Utilizes the `sentence-transformers/all-MiniLM-L6-v2` model for generating high-quality vector embeddings.
*   **Chroma Vector Store:** Stores and retrieves document embeddings for efficient semantic search.
*   **Chat-based Question Answering (RAG):** Integrates a LangChain conversational agent to answer questions using information retrieved from the knowledge base (Retrieval-Augmented Generation).
*   **FAISS LSH Similarity Search:** Demonstrates approximate nearest neighbor search using Locality Sensitive Hashing (LSH) for fast similarity queries. Includes an evaluation of LSH performance with varying `nbits`.
*   **Cosine Similarity Search:** Implements a direct cosine similarity calculation using scikit-learn for precise semantic search.
*   **Interactive t-SNE Visualizations:** Provides 2D and 3D t-SNE plots to visualize the distribution of document embeddings, the query point, and the top search results, color-coded by similarity.
*   **Gradio Interface:** An interactive chat interface built with Gradio for real-time interaction with the LangChain agent.

## 🚀 Getting Started

Follow these steps to set up and run the project locally.

### Prerequisites

*   Python 3.13.2.
*   Git (for cloning the repository).

### Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/YourGitHubUsername/LangChain_HW2.git
    cd LangChain_HW2
    ```
    (Replace `YourGitHubUsername/LangChain_HW2.git` with your actual repository URL)

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    ```

3.  **Activate the virtual environment:**
    *   **Windows:**
        ```bash
        .\venv\Scripts\activate
        ```
    *   **macOS/Linux:**
        ```bash
        source venv/bin/activate
        ```

4.  **Install project dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

5.  **Prepare the knowledge base:**
    You should have received a dataset named `knowledge-base2.zip`.
    *   Extract its contents directly into the project's root directory. Ensure that there is a folder named `knowledge-base2/` in the same directory as your `LangChain_hw2.ipynb` file.

6.  **Set up API Keys (Optional, for external LLM access):**
    The notebook is configured to use `google/gemma-3-27b-it:free` via [OpenRouter.ai](https://openrouter.ai/).
    *   Create a `.env` file in the project's root directory (same level as `LangChain_hw2.ipynb`).
    *   Add your OpenRouter API key (or any other API key you intend to use via OpenRouter) to this file:
        ```
        GOOGLE_API_KEY="sk-or-v1-xxxxxxxxxxxxxxxxxxxxxx" # Replace with your actual OpenRouter API Key
        # LLAMA_API_KEY="sk-or-v1-xxxxxxxxxxxxxxxxxxxxxx" # Example for another OpenRouter model
        # QWEN_API_KEY="sk-or-v1-xxxxxxxxxxxxxxxxxxxxxx" # Example for another OpenRouter model
        # DEEPSEEK_API_KEY="sk-or-v1-xxxxxxxxxxxxxxxxxxxxxx" # Example for another OpenRouter model
        ```
    *   **Alternatively (for local LLM):** If you prefer to use a local Ollama model (e.g., `llama3.2:latest`) as mentioned in the assignment, ensure Ollama is installed and running, and the model is pulled (`ollama pull llama3.2`). You would then uncomment and use the `ChatOllama` line in the notebook (Cell 24). In this case, no API key is strictly required, but defining `ollama` as the `api_key` in LangChain's `OpenAI` client is a common practice for local Ollama instances.

### Running the Project

1.  **Launch Jupyter Notebook with your favorite IDE :)

2.  **Run All Cells:** Execute all cells in the notebook sequentially. This will:
    *   Load and chunk documents.
    *   Create and persist the Chroma vector store.
    *   Initialize the LangChain conversational agent.
    *   Perform FAISS LSH and Cosine Similarity searches.
    *   Generate all visualizations.
    *   Launch the Gradio chat interface (you'll see a local URL in the output, e.g., `http://127.0.0.1:7860/`).

## 📂 Project Structure

```
.
├── knowledge-base2/
│   ├── Analytical Chemistry.txt
│   ├── Artificial Intelligence.txt
│   └── ... (other knowledge base documents)
├── LangChain_hw2.ipynb
├── requirements.txt
└── .env (optional, for API keys)
```

## 📈 Visualizations and Outputs

The Jupyter notebook will generate several key outputs:

### 1. LangChain Chat Outputs
This section demonstrates the RAG-enabled conversational agent answering questions based on the knowledge base. You'll see the questions asked and the generated answers.

![1  LangChain Output](https://github.com/user-attachments/assets/3e902217-0e91-4938-b1d8-13846689916a)


### 2. FAISS LSH Search + t-SNE Plot
This plot visualizes the semantic search results. The query and the top similar documents found (based on the distance metric used in `search_and_visualize`, which is Euclidean distance, and evaluated by FAISS LSH for optimal `nbits`) are highlighted in a 2D t-SNE space, with documents color-coded by their similarity to the query.

![2  FAISS LSH - text output](https://github.com/user-attachments/assets/b4226513-9ff2-4cbe-8c5c-40834d479488)
![2 1 FAISS LSH - Visualization](https://github.com/user-attachments/assets/c670271f-adff-4a5b-8667-116bd264a2a0)


### 3. Cosine Similarity Search + t-SNE Plot
This visualization shows the semantic search results using direct cosine similarity. Similar to the FAISS plot, it highlights the query and top similar documents within the t-SNE reduced space, with documents color-coded by their cosine similarity scores. A 3D version is also included for a more immersive view.

![3 Cosine Similiarty - text output](https://github.com/user-attachments/assets/6eaad84a-732c-48b6-b19f-a1b221988b8b)
![3 1 Cosine Similiarty - Visualization (2d)](https://github.com/user-attachments/assets/44b8a5f3-d233-4eb4-9ec2-c0c5f95fec8d)
![3 2 Cosine Similarty - Visualization (3d)](https://github.com/user-attachments/assets/6da73a2d-2ba6-42cd-9e41-27975ea40477)



## 💡 Important Implementation Notes

*   **Embedding Model:** The project exclusively uses `HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")` as specified.
*   **LLM Choice:** The notebook is set up to use `ChatOpenAI` with `openrouter.ai` and the `google/gemma-3-27b-it:free` model. An alternative `ChatOllama` configuration for local models is commented out.
*   **FAISS LSH:** The FAISS LSH index is primarily used for evaluating optimal `nbits` parameters, demonstrating its role in approximate similarity search. The `search_and_visualize` function then performs a general similarity search (Euclidean distance) across all documents and visualizes the results, providing insights into the embedding space.
*   **Cosine Similarity:** The `cosine_search_and_visualize` and `cosine_search_3d` functions directly compute cosine similarity using `sklearn.metrics.pairwise.cosine_similarity` for visualization.

## 🤝 Contributing

Feel free to fork this repository, open issues, or submit pull requests.



---
