import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import json
import time
from sentence_transformers import SentenceTransformer
import psycopg2
from pgvector.psycopg2 import register_vector
from groq import Groq
import streamlit as st
# from opik import track
from dotenv import load_dotenv

load_dotenv()

db_config = {
    "dbname": os.getenv("DB_NAME"),
    "user": os.getenv("DB_USER"),
    "password": os.getenv("DB_PASSWORD"),
    "host": os.getenv("DB_HOST"),
    "port": os.getenv("DB_PORT")
}

groq_api_key = os.getenv("GROQ_API_KEY")


def classify_query(query, groq_api_key):
    client = Groq(api_key=groq_api_key)
    
    # Updated classification prompt to include greetings, goodbyes, medical, and non-medical categories
    classification_prompt = (
        "You are an expert in classifying queries into one of the following categories:\n"
        "1. 'greeting' - If the query is a greeting like 'hello', 'hi', 'good morning', 'how are you', 'how's the day', 'how's it going' etc.\n"
        "2. 'goodbye' - If the query is a farewell like 'goodbye', 'bye', 'see you', etc.\n"
        "3. 'medical' - If the query relates to diseases, symptoms, treatments, medical procedures, medications, healthcare guidelines, etc.\n"
        "4. 'non-medical' - If the query is unrelated to health or medicine, such as general questions, personal queries, or non-health-related topics.\n\n"
        "5. 'medical' - If the query is ambiguous or could be interpreted in both medical and non-medical contexts, classify it as medical if it has a plausible medical interpretation.\n\n"
        "Classify the following query into one of the above categories and respond in JSON format with a single key 'category'.\n"
        f"Query: {query}\n"
        "Respond only with valid JSON. Example:\n"
        '{"category": "medical"}\n'
        "Answer:"
    )
    
    try:
        # Get the model's response
        response = client.chat.completions.create(
            messages=[{"role": "user", "content": classification_prompt}],
            max_tokens=50,  # Increase tokens to allow for JSON response
            temperature=0.1,
            model="llama-3.1-8b-instant",
        )
        
        # Extract the response content
        response_content = response.choices[0].message.content.strip()
        
        # Debugging: Print the raw response
        print("Raw Groq API Response:", response_content)
        
        # Parse the JSON response
        try:
            response_json = json.loads(response_content)
            category = response_json.get("category", "").lower()
            
            # Return JSON with category and original response
            return {
                "category": category,  # "greeting", "goodbye", "medical", or "non-medical"
                "original_response": response_content  # Include original response for debugging
            }
        except json.JSONDecodeError:
            print("Failed to parse JSON response. Falling back to non-medical.")
            return {
                "category": "non-medical",  # Fallback to non-medical
                "original_response": response_content
            }
    except Exception as e:
        print(f"Error in is_medical_query: {e}")
        return {
            "category": "non-medical",  # Fallback to non-medical
            "original_response": str(e)
        }

# @track
def query_vector_store(query, db_config, top_k=2, model_name="all-MiniLM-L6-v2"):
    model = SentenceTransformer(model_name)
    conn = psycopg2.connect(**db_config)
    register_vector(conn)
    cur = conn.cursor()
    query_embedding = model.encode(query).tolist()
    cur.execute("""
        SELECT md_file, pdf_link, chunk_index, chunk_text
        FROM document_embeddings
        ORDER BY embedding <-> %s::vector
        LIMIT %s;
    """, (query_embedding, top_k))
    results = [{"md_file": row[0], "pdf_link": row[1], "chunk_index": row[2], "chunk_text": row[3]} for row in cur.fetchall()]
    cur.close()
    conn.close()
    return results

# @track
def generate_answer_with_groq(query, retrieved_texts, groq_api_key):
    client = Groq(api_key=groq_api_key)

        #     f"The following texts were retrieved as context:\n\n"
    #     f"{retrieved_texts}\n\n"
    #     f"Based on the context, answer the query:\n\n"
    #     f"{query}"
    #     f"Some rules to follow:\n"
    #     f"Avoid statements like 'Based on the context, ...' or 'The context information ...' or anything along those lines.\n"
    #     f"Never try to add information from yourself always adhere to the provided context information. Avoid to add hallucinations, stay true to the context.\n"
    # )
    
    prompt = (
        f"The following texts were retrieved as context:\n\n"
        f"{retrieved_texts}\n\n"
        f"Based on the context, provide a concise and actionable answer to the query:\n\n"
        f"{query}\n\n"
        f"Rules to follow:\n"
        f"1. Avoid copying text verbatim from the context.\n"
        f"2. Do not include statements like 'Based on the context' or 'The context information'.\n"
        f"3. Structure the response with clear headings whenever possible(e.g., '1. Immediate Actions', '2. Investigations', etc.).\n"
        f"4. Use bullet points for clarity and readability.\n"
        f"5. Never try to add information from yourself always adhere to the provided context information. Avoid to add hallucinations, stay true to the context.\n"
    )
    
    response = client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        max_tokens=1024,
        temperature=0.1,
        model="llama-3.1-8b-instant",
    )
    if "I don't know" in response.choices[0].message.content.lower():
        return "I cannot answer this question from the provided context."
    return response.choices[0].message.content

def main():
    st.title("AI ASSISTANT FOR NICE GUIDELINES")
    st.write("Welcome to the NICE Guidelines assistance")

    if 'history' not in st.session_state:
        st.session_state.history = []
    query = st.text_area("Enter your query to get assistance:", height=150)
    if st.button("Submit"):
        if query:
            query_result = classify_query(query, groq_api_key)
            
            # Debugging: Print the query result
            print("Query Result:", query_result)
            
            category = query_result.get("category", "").lower()
            
            if category == "greeting":
                st.subheader("Response:")
                st.write(f"Hello! How can I assist you today?")
            elif category == "goodbye":
                st.subheader("Response:")
                st.write(f"Goodbye! See you soon. If you need anything else, please ask.")
            elif category == "non-medical":
                st.subheader("Response:")
                st.write("Please ask a relevant query from NICE GUIDELINES.")
            elif category == "medical":
                # db_config = {"dbname": "vector_db", "user": "postgres", "password": "test", "host": "localhost", "port": 5432}
                retrieval_start_time = time.time()
                with st.spinner("Retrieving relevant documents..."):
                    results = query_vector_store(query, db_config)
                    retrieved_texts = "\n\n".join([res["chunk_text"] for res in results])
                    pdf_links = [res["pdf_link"] for res in results]
                retrieval_elapsed_time = time.time() - retrieval_start_time
                st.subheader("Retrieved Context:")
                st.text_area("Context", retrieved_texts, height=300)
                generation_start_time = time.time()
                with st.spinner("Generating answer..."):
                    answer = generate_answer_with_groq(query, retrieved_texts, groq_api_key)
                generation_elapsed_time = time.time() - generation_start_time
                st.subheader("Answer:")
                st.write(answer)
                st.subheader("Elapsed Time:")
                st.write(f"Time for context retrieval: {retrieval_elapsed_time:.2f} seconds")
                st.write(f"Time for answer generation: {generation_elapsed_time:.2f} seconds")
                st.write(f"Total time: {retrieval_elapsed_time + generation_elapsed_time:.2f} seconds")
                if pdf_links:
                    st.subheader("References:")
                    for link in pdf_links:
                        cleaned_link = link.removeprefix("http://localhost:8501/")
                        if not cleaned_link.startswith("http"):
                            cleaned_link = "https://" + cleaned_link
                        st.markdown(f"{cleaned_link}")
                st.session_state.history.append({"query": query, "answer": answer, "retrieval_time": retrieval_elapsed_time, "generation_time": generation_elapsed_time, "total_time": retrieval_elapsed_time + generation_elapsed_time})
            else:
                st.subheader("Response:")
                st.write("Unable to determine the type of query. Please try again.")
    st.sidebar.title("Interaction History")
    if st.session_state.history:
        for interaction in st.session_state.history:
            with st.sidebar.expander(f"**Question :** {interaction['query']}"):
                st.write(f"**Answer :** {interaction['answer']}")
                st.write(f"**Time for Context Retrieval:** {interaction['retrieval_time']:.2f} seconds")
                st.write(f"**Time for Answer Generation:** {interaction['generation_time']:.2f} seconds")
                st.write(f"**Total Time:** {interaction['total_time']:.2f} seconds")

if __name__ == "__main__":
    main()
