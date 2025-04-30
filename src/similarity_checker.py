import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# Load Sentence Transformer model for similarity detection
model = SentenceTransformer('all-mpnet-base-v2')

# Load T5 model for summarization
summarizer_model = AutoModelForSeq2SeqLM.from_pretrained("t5-small")
tokenizer = AutoTokenizer.from_pretrained("t5-small")

# Function to get weighted document embedding
def get_weighted_document_embedding(doc, model, chunk_size=100):
    words = doc.split()  # Split by whitespace
    chunks = [" ".join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]

    embeddings = model.encode(chunks)  # Get chunk embeddings
    weights = np.linspace(1, 2, len(embeddings))  # Assign increasing weights
    weighted_embedding = np.average(embeddings, axis=0, weights=weights)

    return weighted_embedding, chunks, embeddings  # Return embeddings & original chunks

# Function to extract key overlapping content
def extract_overlapping_content(doc1_chunks, doc1_embeddings, doc2_chunks, doc2_embeddings, top_n=5):
    similarities = cosine_similarity(doc1_embeddings, doc2_embeddings)  # Compare all chunks
    top_indices = np.unravel_index(np.argsort(similarities.ravel())[-top_n:], similarities.shape)  # Get top matches

    overlapping_text = []
    for i, j in zip(top_indices[0], top_indices[1]):
        overlapping_text.append(doc1_chunks[i])

    return "\n".join(overlapping_text)  # Return joined similar sections

# Function to summarize overlapping content
def summarize_similarity(overlapping_content):
    if not overlapping_content.strip():
        return "There is no significant overlap between the two documents."

    prompt = "Summarize the following overlapping research content:\n" + overlapping_content
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    summary_ids = summarizer_model.generate(inputs.input_ids, max_length=100, num_beams=5, early_stopping=True)

    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)

# Function to compare two documents and generate a meaningful summary
def compare_and_summarize(doc1, doc2):
    emb1, doc1_chunks, doc1_embeddings = get_weighted_document_embedding(doc1, model)
    emb2, doc2_chunks, doc2_embeddings = get_weighted_document_embedding(doc2, model)

    # Compute document-level similarity
    similarity_score = cosine_similarity([emb1], [emb2])[0][0]

    # Extract overlapping content
    mutual_content = extract_overlapping_content(doc1_chunks, doc1_embeddings, doc2_chunks, doc2_embeddings)

    # Summarize the mutual content
    summarized_overlap = summarize_similarity(mutual_content)

    # Generate final report
    summary = f"🔍 **Similarity Score:** {similarity_score:.4f}\n\n"
    if similarity_score > 0.75:
        summary += "✅ **Your submitted research is highly similar to an existing paper.**\n"
        summary += "📄 **Summary of the overlap:**\n" + summarized_overlap
        summary += "\n⚠️ **Consider revising your research to provide new insights.**"
    elif similarity_score > 0.50:
        summary += "⚠️ **Your submission has some overlapping content with prior research.**\n"
        summary += "📄 **Summary of the overlap:**\n" + summarized_overlap
        summary += "\n✔️ **You might want to refine your approach to introduce more unique aspects.**"
    else:
        summary += "✔️ **Your research appears unique with minimal overlap.**\n"
        summary += "🎉 **Proceed with confidence!**"

    return summary