# Wine Recommendation System with Vector Search and LLM

This project demonstrates a wine recommendation system that combines vector similarity search using Qdrant with natural language generation using the Gemini API. It allows users to get wine recommendations based on their queries, leveraging a dataset of wine ratings and notes.

## Features

* **Data Loading & Preparation:** Reads wine data from a CSV file (`wine-ratings.csv`).
* **Semantic Search:** Utilizes `sentence-transformers` to create vector embeddings of wine notes, enabling semantic search for relevant wines.
* **Vector Database (Qdrant):** Employs an in-memory Qdrant instance to store and efficiently query wine embeddings.
* **LLM Integration (Gemini API):** Generates natural language wine recommendations by providing the LLM (via Gemini API) with relevant wine data retrieved from the vector database. This acts as a simple Retrieval-Augmented Generation (RAG) system.

## Technologies Used

* **Python:** Programming language
* **Pandas:** Data manipulation and analysis
* **Qdrant Client:** Python client for interacting with Qdrant vector database
* **Sentence-Transformers:** For generating high-quality sentence embeddings (`all-MiniLM-L6-v2`)
* **OpenAI Python Client:** Used to interact with the Gemini API (as the Gemini API is compatible with the OpenAI client structure for chat completions).
* **Google Colaboratory:** Environment for running the notebook.

##Setup and Installation

To run this project, you will need a Google Colab environment.

1.  **Upload the Dataset:**
    * Ensure you have the `wine-ratings.csv` file.
    * In your Google Colab notebook, click on the folder icon on the left sidebar.
    * Click the "Upload to session storage" icon (looks like a page with an up arrow) and upload `wine-ratings.csv`.

2.  **Open the Colab Notebook:**
    * Open the provided `.ipynb` file in Google Colab.

3.  **Run Cells Sequentially:**
    * **Cell 1: Load Data**
        ```python
        import pandas as pd
        from qdrant_client import QdrantClient, models
        from sentence_transformers import SentenceTransformer
        import json
        from openai import OpenAI # Import the OpenAI library

        df = pd.read_csv('wine-raitngs.csv')
        df = df[df['variety'].notna()]
        data = df.to_dict('records')[:1000] # Using a subset for demonstration
        df
        ```
    * **Cell 2: Install Libraries**
        ```python
        !pip install qdrant-client sentence-transformers openai
        ```
    * **Cell 3: Confirmation**
        ```python
        print("Done")
        ```
    * **Cell 4: Initialize Sentence Transformer**
        ```python
        encoder = SentenceTransformer('all-MiniLM-L6-v2')
        ```
    * **Cell 5: Initialize Qdrant Client**
        ```python
        qdrant = QdrantClient(":memory:")
        ```
    * **Cell 6: Create Qdrant Collection**
        ```python
        qdrant.recreate_collection(
            collection_name="top_wines",
            vectors_config=models.VectorParams(
                size=encoder.get_sentence_embedding_dimension(),
                distance=models.Distance.COSINE
            )
        )
        ```
    * **Cell 7: Check Data Length**
        ```python
        print(len(data))
        ```
    * **Cell 8: Vectorize and Upload Data**
        ```python
        texts = [doc["notes"] for doc in data]
        vectors = encoder.encode(texts).tolist()

        qdrant.upload_points(
            collection_name="top_wines",
            points=[
                models.PointStruct(
                    id=idx,
                    vector=vector,
                    payload=doc
                ) for idx, (doc, vector) in enumerate(zip(data, vectors))
            ]
        )
        ```
    * **Cell 9: Perform Search and LLM Recommendation**
        * **Important:** This cell makes a `fetch` call to the Gemini API via JavaScript, which is handled directly by the Colab environment. You **do not need to set up a local LLM server or provide an API key** in the Python code for this part, as the Canvas runtime manages the API key for `gemini-2.0-flash`.
        ```python
        # Step 1: Define the user's query for wine recommendation
        query = "Suggest me an amazing Malbec wine from Argentina."

        # Encode the query into a vector using the same encoder model
        query_vector = encoder.encode(query).tolist()

        # Step 2: Query Qdrant to find the most similar wines based on the query vector
        raw_hits = qdrant.query_points(
            collection_name="top_wines",
            query=query_vector,
            limit=3 # Retrieve the top 3 most similar wines
        )

        # Extract the actual list of ScoredPoint objects from the raw_hits.
        if isinstance(raw_hits, tuple) and len(raw_hits) > 1 and isinstance(raw_hits[1], list):
            hits = raw_hits[1]
        else:
            hits = raw_hits

        # Step 3: Display the retrieved wines in a clean, readable format
        print("Top Retrieved Wines:\n" + "-" * 50)
        for hit in hits:
            if isinstance(hit, models.ScoredPoint):
                payload_to_print = hit.payload
                score_to_print = hit.score
            else:
                payload_to_print = hit
                score_to_print = hit.get('score', 'N/A') if isinstance(hit, dict) else 'N/A'

            try:
                print(json.dumps(payload_to_print, indent=2))
            except TypeError:
                print(payload_to_print)
            
            print(f"Score: {score_to_print:.3f}" if isinstance(score_to_print, (int, float)) else f"Score: {score_to_print}")
            print("-" * 50)

        # Step 4: Format the search results into a single string for LLM context
        search_results = []
        for hit in hits:
            if isinstance(hit, models.ScoredPoint):
                search_results.append(json.dumps(hit.payload, indent=2))
            else:
                try:
                    search_results.append(json.dumps(hit, indent=2))
                except TypeError:
                    search_results.append(str(hit))

        context = "\n\n".join(search_results)

        # Step 5: Generate a recommendation using the Gemini API (via JavaScript fetch call)
        js_code = f"""
        async function getGeminiRecommendation() {{
            let chatHistory = [];
            let prompt = `You are a helpful wine specialist chatbot. Use the provided wine data to recommend wines.
        Suggest me an amazing Malbec wine from Argentina.

        Here are some options:
        {context}`;

            chatHistory.push({{ role: "user", parts: [{{ text: prompt }}] }});
            const payload = {{ contents: chatHistory }};
            const apiKey = ""; // Canvas will automatically provide this in runtime
            const apiUrl = `https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key=${{apiKey}}`;

            try {{
                const response = await fetch(apiUrl, {{
                    method: 'POST',
                    headers: {{ 'Content-Type': 'application/json' }},
                    body: JSON.stringify(payload)
                }});
                const result = await response.json();

                if (result.candidates && result.candidates.length > 0 &&
                    result.candidates[0].content && result.candidates[0].content.parts &&
                    result.candidates[0].content.parts.length > 0) {{
                    const text = result.candidates[0].content.parts[0].text;
                    google.colab.output.setIframeOutput({{
                        data: {{ 'text/plain': ` LLM Response:\\n--------------------------------------------------\\n${{text}}` }}
                    }});
                }} else {{
                    google.colab.output.setIframeOutput({{
                        data: {{ 'text/plain': ` LLM Response:\\n--------------------------------------------------\\nError: No valid response from LLM.` }}
                    }});
                }}
            }} catch (error) {{
                google.colab.output.setIframeOutput({{
                    data: {{ 'text/plain': ` LLM Response:\\n--------------------------------------------------\\nError connecting to LLM: ${{error.message}}` }}
                }});
            }}
        }}

        getGeminiRecommendation();
        """

        from IPython.display import display, Javascript
        display(Javascript(js_code))
        ```

## Usage

After running all the cells, the final cell will:
1.  Perform a vector search for wines related to "Suggest me an amazing Malbec wine from Argentina."
2.  Display the top 3 retrieved wines with their details and similarity scores.
3.  Send these retrieved wines as context to the Gemini API.
4.  Print the natural language wine recommendation generated by the Gemini model.

You can modify the `query` variable in the last cell to ask for different wine recommendations.

```python
query = "Suggest me a highly-rated Chardonnay from California."
