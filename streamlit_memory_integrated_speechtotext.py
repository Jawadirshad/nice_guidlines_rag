# import os
# from typing import List, TypedDict
# os.environ["TOKENIZERS_PARALLELISM"] = "false"

# import json
# import time
# from sentence_transformers import SentenceTransformer
# import psycopg2
# from pgvector.psycopg2 import register_vector
# from psycopg2.pool import SimpleConnectionPool
# import streamlit as st
# from dotenv import load_dotenv
# from openai import OpenAI
# import asyncio
# from concurrent.futures import ThreadPoolExecutor


# # Load environment variables
# load_dotenv()

# db_config = {
#     "dbname": os.getenv("DB_NAME"),
#     "user": os.getenv("DB_USER"),
#     "password": os.getenv("DB_PASSWORD"),
#     "host": os.getenv("DB_HOST"),
#     "port": os.getenv("DB_PORT")
# }

# CONNECTION_POOL = SimpleConnectionPool(
#     minconn=1,
#     maxconn=10,
#     **db_config
# )

# SENTENCE_MODEL = SentenceTransformer("all-MiniLM-L6-v2")
# openai_api_key = os.getenv("OPENAI_API_KEY")
# MODEL = "gpt-4o-mini"

# async def classify_query(query):
#     client = OpenAI(api_key=openai_api_key)
    
#     classification_prompt = (
#         "You are an expert in classifying queries into one of the following categories:\n"
#         "1. 'greeting' - If the query is a greeting like 'hello', 'hi', 'good morning', etc.\n"
#         "2. 'goodbye' - If the query is a farewell like 'goodbye', 'bye', etc.\n"
#         "3. 'medical' - If the query relates to medical topics, diseases, medications, symptoms\n"
#         "4. 'non-medical' - For non-health related topics, riddles queries, gibberish\n"
#         "5. 'medical' - For ambiguous cases in which you can not decide if its non-medical then default it to medical\n\n"

#         f"{query}\n"
#         "Respond only with valid JSON. Example: {'category': 'medical'}"
#     )
    
#     try:
#         print("Sending classification request...")
#         response = await asyncio.to_thread(
#             client.chat.completions.create,
#             messages=[{"role": "user", "content": classification_prompt}],
#             max_tokens=50,
#             temperature=0.1,
#             model=MODEL,
#         )
        
#         response_content = response.choices[0].message.content.strip()
#         print(f"Raw classification response: {response_content}")
#         response_content = response_content.replace("```json", "").replace("```", "").strip()
#         response_content = response_content.replace("'", '"')
        
#         try:
#             response_json = json.loads(response_content)
#             category = response_json.get("category", "").lower()
#             print(f"Parsed category: {category}")
#             return {"category": category}
#         except json.JSONDecodeError:
#             return {"category": "medical"}
#     except Exception as e:
#         return {"category": "medical"}

# def retrieve(query: str):
#     """Retrieve information related to a query."""
#     print(f"\n=== RETRIEVAL PHASE ===")
#     print(f"Query: '{query}'")
#     try:
#         conn = CONNECTION_POOL.getconn()
#         register_vector(conn)
#         cur = conn.cursor()
        
#         print("Generating embeddings...")
#         query_embedding = SENTENCE_MODEL.encode(query).tolist()
        
#         print("Executing similarity search...")
#         cur.execute("""
#             SELECT md_file, pdf_link, chunk_index, chunk_text
#             FROM document_embeddings
#             ORDER BY embedding <-> %s::vector
#             LIMIT 2;
#         """, (query_embedding,))
        
#         results = cur.fetchall()
#         print(f"Found {len(results)} matching chunks")
        
#         formatted_results = [{
#             "md_file": row[0],
#             "pdf_link": row[1],
#             "chunk_index": row[2],
#             "chunk_text": row[3]
#         } for row in results]
        
#         cur.close()
#         CONNECTION_POOL.putconn(conn)
        
#         serialized = "\n\n".join(doc["chunk_text"] for doc in formatted_results)
#         print("Retrieval completed successfully")
#         return json.dumps({"content": serialized, "results": formatted_results})
#     except Exception as e:
#         print(f"!! Retrieval error: {str(e)}")
#         return json.dumps({"content": "", "results": []})

# async def run_rag_conversation(query: str, conversation_history: list):
#     print(f"\n=== RAG PROCESSING ===")
#     client = OpenAI(api_key=openai_api_key)

#     system_prompt = """You MUST follow these rules:
#     1. FIRST check the provided context THOROUGHLY. If the user's query already exists or is completely answered in the context, respond using ONLY that context WITHOUT retrieval.
#     2. If answer is not found in provided context, ALWAYS trigger tool call to retrieve from NICE Guidelines database
#     3. NEVER respond using your own internal knowledge outside these sources(tool call, provided context)
#     4. Tool Calls:
#        - When rephrasing the user query try to keep the rephrased queries of medium size and include all information of user query.
#        - Never create more than two rephrased queries
#        - Include relevant context from the user query for better retrieval.
#     5. Final Responses:
#        - Do not include statements like 'Based on the context' or 'The context information' or 'According to the context'.\n"
#        - Do not cite any links or resources in the final response.
#        - Structure the response with clear headings whenever possible(e.g., '1. Immediate Actions', '2. Investigations', etc.).\n"
#        - Use bullet points for clarity and readability.\n"""  
    
#     tools = [
#         {
#             "type": "function",
#             "function": {
#                 "name": "retrieve",
#                 "description": "Retrieve relevant information from the NICE Guidelines database, the retrieval is must unless answer is fully present in provided context",
#                 "parameters": {
#                     "type": "object",
#                     "properties": {
#                         "query": {
#                             "type": "string",
#                             "description": "The search query to retrieve information",
#                         }
#                     },
#                     "required": ["query"],
#                 }
#             }
#         }
#     ]
    
#     messages = [{"role": "system", "content": system_prompt}] + conversation_history + [{"role": "user", "content": query}]

#     response = await asyncio.to_thread(
#         client.chat.completions.create,
#         model=MODEL,
#         messages=messages,
#         tools=tools,
#         tool_choice="auto",
#     )
    
#     response_message = response.choices[0].message
#     tool_calls = response_message.tool_calls
#     messages.append(response_message)
    
#     retrieved_results = []
#     context = None
    
#     if tool_calls:
#         print(f"2. Tool Calls Detected ({len(tool_calls)})")
#         # Handle multiple tool calls in parallel
#         with ThreadPoolExecutor(max_workers=3) as executor:
#             futures = []
#             for tool_call in tool_calls:
#                 if tool_call.function.name == "retrieve":
#                     function_args = json.loads(tool_call.function.arguments)
#                     futures.append(executor.submit(retrieve, function_args.get("query")))
            
#             # Collect results
#             for tool_call, future in zip(tool_calls, futures):
#                 function_response = future.result()
#                 retrieved_data = json.loads(function_response)
#                 retrieved_results.extend(retrieved_data.get("results", []))
#                 messages.append({
#                     "tool_call_id": tool_call.id,
#                     "role": "tool",
#                     "name": "retrieve",
#                     "content": function_response,
#                 })
#                 if not context:  # Use the first non-empty context
#                     context = retrieved_data.get("content", "")

#     final_response = await asyncio.to_thread(
#         client.chat.completions.create,
#         model=MODEL,
#         messages=messages
#     )

#     answer = final_response.choices[0].message.content
#     print("5. Response Generation Complete")
    
#     return {
#         "answer": answer,
#         "context": context,
#         "results": retrieved_results
#     }

# async def main_async():
#     st.title("AI ASSISTANT FOR NICE GUIDELINES")
#     st.write("Welcome to the NICE Guidelines assistance")

#     if 'history' not in st.session_state:
#         st.session_state.history = []
#     if 'conversation_history' not in st.session_state:
#         st.session_state.conversation_history = []

#     st.subheader("Session History")
#     if st.session_state.history:
#         for interaction in st.session_state.history[::-1]: 
#             with st.expander(f"Query: {interaction['query']}"):
#                 st.write(f"Response: {interaction['answer']}")
#                 st.caption(f"Processing time: {interaction['time']:.2f}s")

#     # Input box for new query
#     st.subheader("Ask Your Question")
#     query = st.text_area("Enter your query:", height=150)

#     # Submit button
#     if st.button("Submit"):
#         if query:
#             print("\n" + "="*60)
#             print(f"NEW QUERY: '{query}'")
#             start_time = time.time()

#             # Classification
#             classification_start = time.time()
#             query_result = await classify_query(query)
#             category = query_result.get("category", "").lower()
#             print(f"\nClassification: {category} (took {time.time() - classification_start:.2f}s)")

#             if category in ["greeting", "goodbye", "non-medical"]:
#                 print("Using template response")
#                 responses = {
#                     "greeting": "Hello! How can I assist you today?",
#                     "goodbye": "Goodbye! Feel free to ask more questions anytime.",
#                     "non-medical": "Please ask a question related to NICE Guidelines."
#                 }
#                 response = responses[category]

#                 st.subheader("Response:")
#                 st.write(response)

#             elif category == "medical":
#                 print("Initiating RAG pipeline")
#                 rag_start = time.time()

#                 with st.spinner("Analyzing your query..."):
#                     result = await run_rag_conversation(query, st.session_state.conversation_history)

#                     # Existing context display
#                     if result["context"]:
#                         print(f"Context length: {len(result['context'])} characters")
#                         st.subheader("Relevant Guidelines:")
#                         st.text_area("Context", result["context"], height=300)

#                     # Existing answer display
#                     st.subheader("Expert Analysis:")
#                     st.write(result["answer"])

#                     # New PDF links display
#                     if result["results"]:
#                         st.subheader("References:")
#                         # Get unique PDF links with cleaning
#                         pdf_links = list({res["pdf_link"].replace("http://localhost:8501/", "") 
#                                         for res in result["results"] if res.get("pdf_link")})
                        
#                         for link in pdf_links:
#                             # Ensure valid URL format
#                             if not link.startswith("http"):
#                                 link = f"https://{link}"
#                             st.markdown(f"[{link}]({link})")

#                 rag_time = time.time() - rag_start
#                 print(f"RAG processing took {rag_time:.2f} seconds")

#                 max_history = 8 * 2
#                 # Add new messages to conversation history
#                 new_messages = [
#                     {"role": "user", "content": query},
#                     {"role": "assistant", "content": result["answer"]}
#                 ]

#                 if len(st.session_state.conversation_history) + len(new_messages) > max_history:
#                     # Calculate how many messages to remove (oldest first)
#                     excess = (len(st.session_state.conversation_history) + len(new_messages)) - max_history
#                     st.session_state.conversation_history = st.session_state.conversation_history[excess:]

                
#                 st.session_state.conversation_history.extend([
#                     {"role": "user", "content": query},
#                     {"role": "assistant", "content": result["answer"]}
#                 ])

#                 st.session_state.history.append({
#                     "query": query,
#                     "answer": result["answer"],
#                     "time": time.time() - start_time
#                 })


# def main():
#     asyncio.run(main_async())

# if __name__ == "__main__":
#     main()



# making speech to text changes
import os
from typing import List, TypedDict
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import json
import time
import tempfile
import numpy as np
import sounddevice as sd
import soundfile as sf
from sentence_transformers import SentenceTransformer
import psycopg2
from pgvector.psycopg2 import register_vector
from psycopg2.pool import SimpleConnectionPool
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI
import asyncio
from concurrent.futures import ThreadPoolExecutor


# Load environment variables
load_dotenv()

# Initialize session state variables
# Initialize session state variables
if 'history' not in st.session_state:
    st.session_state.history = []
if 'conversation_history' not in st.session_state:
    st.session_state.conversation_history = []
if 'recording' not in st.session_state:
    st.session_state.recording = False
if 'query_text' not in st.session_state:
    st.session_state.query_text = ""

db_config = {
    "dbname": os.getenv("DB_NAME"),
    "user": os.getenv("DB_USER"),
    "password": os.getenv("DB_PASSWORD"),
    "host": os.getenv("DB_HOST"),
    "port": os.getenv("DB_PORT")
}

CONNECTION_POOL = SimpleConnectionPool(
    minconn=1,
    maxconn=10,
    **db_config
)

SENTENCE_MODEL = SentenceTransformer("all-MiniLM-L6-v2")
openai_api_key = os.getenv("OPENAI_API_KEY")
MODEL = "gpt-4o-mini"
client = OpenAI(api_key=openai_api_key)

# Audio recording settings
SAMPLE_RATE = 16000
CHANNELS = 1
DTYPE = 'float32'

def record_audio(duration=10):
    """Record audio for a specified duration"""
    try:
        # Calculate number of frames
        frames = int(SAMPLE_RATE * duration)
        
        # Create progress bar
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Initialize recording array
        recording = np.zeros((frames, CHANNELS), dtype=DTYPE)
        
        # Start recording
        with sd.InputStream(samplerate=SAMPLE_RATE, channels=CHANNELS, dtype=DTYPE) as stream:
            for i in range(duration):
                if not st.session_state.recording:
                    break
                
                # Update progress
                progress_bar.progress((i + 1) / duration)
                status_text.text(f"Recording... {duration - i} seconds remaining")
                
                # Record chunk
                chunk, overflowed = stream.read(SAMPLE_RATE)
                if overflowed:
                    st.warning("Audio buffer has overflowed")
                
                # Save chunk to recording array
                start_idx = i * SAMPLE_RATE
                end_idx = start_idx + len(chunk)
                recording[start_idx:end_idx] = chunk
        
        # Cleanup UI elements
        progress_bar.empty()
        status_text.empty()
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
            sf.write(tmp_file.name, recording, SAMPLE_RATE)
            return tmp_file.name
            
    except Exception as e:
        st.error(f"Error recording audio: {str(e)}")
        return None
    finally:
        st.session_state.recording = False

def process_audio(audio_file_path):
    """Process recorded audio using OpenAI Whisper API"""
    try:
        client = OpenAI(api_key=openai_api_key)
        
        with open(audio_file_path, 'rb') as audio_file:
            transcript = client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file
            )
        
        # Clean up temporary file
        os.unlink(audio_file_path)
        
        return transcript.text
    except Exception as e:
        st.error(f"Error in audio processing: {str(e)}")
        return None

async def classify_query(query):
    classification_prompt = (
        "You are an expert in classifying queries into one of the following categories:\n"
        "1. 'greeting' - If the query is a greeting like 'hello', 'hi', 'good morning', etc.\n"
        "2. 'goodbye' - If the query is a farewell like 'goodbye', 'bye', etc.\n"
        "3. 'medical' - If the query relates to medical topics, diseases, medications, symptoms\n"
        "4. 'non-medical' - For non-health related topics, riddles queries, gibberish\n"
        "5. 'medical' - For ambiguous cases in which you can not decide if its non-medical then default it to medical\n\n"

        f"{query}\n"
        "Respond only with valid JSON. Example: {'category': 'medical'}"
    )
    
    try:
        print("Sending classification request...")
        response = await asyncio.to_thread(
            client.chat.completions.create,
            messages=[{"role": "user", "content": classification_prompt}],
            max_tokens=50,
            temperature=0.1,
            model=MODEL,
        )
        
        response_content = response.choices[0].message.content.strip()
        print(f"Raw classification response: {response_content}")
        response_content = response_content.replace("```json", "").replace("```", "").strip()
        response_content = response_content.replace("'", '"')
        
        try:
            response_json = json.loads(response_content)
            category = response_json.get("category", "").lower()
            print(f"Parsed category: {category}")
            return {"category": category}
        except json.JSONDecodeError:
            return {"category": "medical"}
    except Exception as e:
        return {"category": "medical"}

def retrieve(query: str):
    """Retrieve information related to a query."""
    print(f"\n=== RETRIEVAL PHASE ===")
    print(f"Query: '{query}'")
    try:
        conn = CONNECTION_POOL.getconn()
        register_vector(conn)
        cur = conn.cursor()
        
        print("Generating embeddings...")
        query_embedding = SENTENCE_MODEL.encode(query).tolist()
        
        print("Executing similarity search...")
        cur.execute("""
            SELECT md_file, pdf_link, chunk_index, chunk_text
            FROM document_embeddings
            ORDER BY embedding <-> %s::vector
            LIMIT 2;
        """, (query_embedding,))
        
        results = cur.fetchall()
        print(f"Found {len(results)} matching chunks")
        
        formatted_results = [{
            "md_file": row[0],
            "pdf_link": row[1],
            "chunk_index": row[2],
            "chunk_text": row[3]
        } for row in results]
        
        cur.close()
        CONNECTION_POOL.putconn(conn)
        
        serialized = "\n\n".join(doc["chunk_text"] for doc in formatted_results)
        print("Retrieval completed successfully")
        return json.dumps({"content": serialized, "results": formatted_results})
    except Exception as e:
        print(f"!! Retrieval error: {str(e)}")
        return json.dumps({"content": "", "results": []})

async def run_rag_conversation(query: str, conversation_history: list):
    print(f"\n=== RAG PROCESSING ===")
    client = OpenAI(api_key=openai_api_key)

    system_prompt = """You MUST follow these rules:
    1. FIRST check the provided context THOROUGHLY. If the user's query already exists or is completely answered in the context, respond using ONLY that context WITHOUT retrieval.
    2. If answer is not found in provided context, ALWAYS trigger tool call to retrieve from NICE Guidelines database
    3. NEVER respond using your own internal knowledge outside these sources(tool call, provided context)
    4. Tool Calls:
       - When rephrasing the user query try to keep the rephrased queries of medium size and include all information of user query.
       - Never create more than two rephrased queries
       - Include relevant context from the user query for better retrieval.
    5. Final Responses:
       - Do not include statements like 'Based on the context' or 'The context information' or 'According to the context'.\n"
       - Do not cite any links or resources in the final response.
       - Structure the response with clear headings whenever possible(e.g., '1. Immediate Actions', '2. Investigations', etc.).\n"
       - Use bullet points for clarity and readability.\n"""  
    
    tools = [
        {
            "type": "function",
            "function": {
                "name": "retrieve",
                "description": "Retrieve relevant information from the NICE Guidelines database, the retrieval is must unless answer is fully present in provided context",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The search query to retrieve information",
                        }
                    },
                    "required": ["query"],
                }
            }
        }
    ]
    
    messages = [{"role": "system", "content": system_prompt}] + conversation_history + [{"role": "user", "content": query}]

    response = await asyncio.to_thread(
        client.chat.completions.create,
        model=MODEL,
        messages=messages,
        tools=tools,
        tool_choice="auto",
    )
    
    response_message = response.choices[0].message
    tool_calls = response_message.tool_calls
    messages.append(response_message)
    
    retrieved_results = []
    context = None
    
    if tool_calls:
        print(f"2. Tool Calls Detected ({len(tool_calls)})")
        # Handle multiple tool calls in parallel
        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = []
            for tool_call in tool_calls:
                if tool_call.function.name == "retrieve":
                    function_args = json.loads(tool_call.function.arguments)
                    futures.append(executor.submit(retrieve, function_args.get("query")))
            
            # Collect results
            for tool_call, future in zip(tool_calls, futures):
                function_response = future.result()
                retrieved_data = json.loads(function_response)
                retrieved_results.extend(retrieved_data.get("results", []))
                messages.append({
                    "tool_call_id": tool_call.id,
                    "role": "tool",
                    "name": "retrieve",
                    "content": function_response,
                })
                if not context:  # Use the first non-empty context
                    context = retrieved_data.get("content", "")

    final_response = await asyncio.to_thread(
        client.chat.completions.create,
        model=MODEL,
        messages=messages
    )

    answer = final_response.choices[0].message.content
    print("5. Response Generation Complete")
    
    return {
        "answer": answer,
        "context": context,
        "results": retrieved_results
    }

async def main_async():
    st.title("AI ASSISTANT FOR NICE GUIDELINES")
    st.write("Welcome to the NICE Guidelines assistance")

    if 'history' not in st.session_state:
        st.session_state.history = []
    if 'conversation_history' not in st.session_state:
        st.session_state.conversation_history = []

    st.subheader("Session History")
    if st.session_state.history:
        for interaction in st.session_state.history[::-1]: 
            with st.expander(f"Query: {interaction['query']}"):
                st.write(f"Response: {interaction['answer']}")
                st.caption(f"Processing time: {interaction['time']:.2f}s")

    # Input box for new query
    st.subheader("Ask Your Question")
    # query = st.text_area("Enter your query:", height=150)

    # Create two columns for the input area
    col1, col2 = st.columns([4, 1])
    
    with col1:
        query = st.text_area(
            "Enter your query:",
            value=st.session_state.query_text,
            height=150,
            key="query_input"
        )

    with col2:
        st.write("")  # Add some spacing
        st.write("")  # Add some spacing
        
        # Audio recording button
        if not st.session_state.recording:
            if st.button("🎤 Start Recording"):
                st.session_state.recording = True
                audio_file = record_audio()
                if audio_file:
                    with st.spinner("Processing audio..."):
                        transcription = process_audio(audio_file)
                        if transcription:
                            st.session_state.query_text = transcription
                            st.rerun()
        else:
            if st.button("⏹️ Stop Recording"):
                st.session_state.recording = False

    # Submit button
    if st.button("Submit"):
        if query:
            print("\n" + "="*60)
            print(f"NEW QUERY: '{query}'")
            start_time = time.time()

            # Classification
            classification_start = time.time()
            query_result = await classify_query(query)
            category = query_result.get("category", "").lower()
            print(f"\nClassification: {category} (took {time.time() - classification_start:.2f}s)")

            if category in ["greeting", "goodbye", "non-medical"]:
                print("Using template response")
                responses = {
                    "greeting": "Hello! How can I assist you today?",
                    "goodbye": "Goodbye! Feel free to ask more questions anytime.",
                    "non-medical": "Please ask a question related to NICE Guidelines."
                }
                response = responses[category]

                st.subheader("Response:")
                st.write(response)

            elif category == "medical":
                print("Initiating RAG pipeline")
                rag_start = time.time()

                with st.spinner("Analyzing your query..."):
                    result = await run_rag_conversation(query, st.session_state.conversation_history)

                    # Existing context display
                    if result["context"]:
                        print(f"Context length: {len(result['context'])} characters")
                        st.subheader("Relevant Guidelines:")
                        st.text_area("Context", result["context"], height=300)

                    # Existing answer display
                    st.subheader("Expert Analysis:")
                    st.write(result["answer"])

                    # New PDF links display
                    if result["results"]:
                        st.subheader("References:")
                        # Get unique PDF links with cleaning
                        pdf_links = list({res["pdf_link"].replace("http://localhost:8501/", "") 
                                        for res in result["results"] if res.get("pdf_link")})
                        
                        for link in pdf_links:
                            # Ensure valid URL format
                            if not link.startswith("http"):
                                link = f"https://{link}"
                            st.markdown(f"[{link}]({link})")

                rag_time = time.time() - rag_start
                print(f"RAG processing took {rag_time:.2f} seconds")

                max_history = 8 * 2
                # Add new messages to conversation history
                new_messages = [
                    {"role": "user", "content": query},
                    {"role": "assistant", "content": result["answer"]}
                ]

                if len(st.session_state.conversation_history) + len(new_messages) > max_history:
                    # Calculate how many messages to remove (oldest first)
                    excess = (len(st.session_state.conversation_history) + len(new_messages)) - max_history
                    st.session_state.conversation_history = st.session_state.conversation_history[excess:]

                
                st.session_state.conversation_history.extend([
                    {"role": "user", "content": query},
                    {"role": "assistant", "content": result["answer"]}
                ])

                st.session_state.history.append({
                    "query": query,
                    "answer": result["answer"],
                    "time": time.time() - start_time
                })


def main():
    asyncio.run(main_async())

if __name__ == "__main__":
    main()