---

# 🚀 Multimodal Similarity Search

This system performs **semantic similarity analysis** between audio, visual, and textual content by converting them into vector embeddings.
It enables **content-based similarity detection**, which can be leveraged for **advertisement targeting, content moderation, or personalized content recommendation** on platforms dealing with large-scale image, video, or audio data.

---

### **Workflow Breakdown**

1. ** Video and Frame Extraction**

   * Frames are extracted from videos using **OpenCV** at defined intervals or scene boundaries.
   * This step isolates key visuals for downstream processing.

2. ** Text Generation**

   * Extracted frames are passed through **BLIP** (Bootstrapping Language-Image Pretraining), an image captioning model, to generate descriptive textual representations.
   * This converts visual content into natural language for semantic comparison.

3. ** Embedding Generation**

   * Captions are embedded using a **sentence-transformer model** (e.g., BERT).
   * Each caption is transformed into a dense numerical vector encoding semantic meaning.

4. ** Similarity Score**

   * **Cosine similarity** is computed between the two embedding vectors.
   * The output score (range 0–1) quantifies how contextually similar the two inputs are.

5. ** Audio Pipeline**

   * An **Automatic Speech Recognition (ASR)** model generates transcripts from audio inputs.
   * A **fine-tuned Whisper model** optimized for Indian native languages is used to improve transcription accuracy (WER).
   * The rest of the embedding and similarity workflow mirrors the image pipeline.

---

## ⚙️ Setup Instructions

### 1. Clone the repository

```bash
git clone https://github.com/your-username/similarity-api.git
cd similarity-api
```

### 2. Create a virtual environment & install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the FastAPI server

```bash
uvicorn app:app --reload
```
