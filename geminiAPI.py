from google import genai
from google.genai import types
from credentials import api_key
def evaluate_query(expected_query, actual_query): 
    # Evaluate semantic similarity between expected and actual queries
    client = genai.Client(
        api_key=api_key
    )
    
    system_prompt = """You are an expert at analyzing semantic similarity between queries.
    Compare the following queries and determine if they have the same meaning/intent.
    Respond only with '1' if similar or '0' if not."""
    
    prompt = f"""Query 1: {expected_query}
    Query 2: {actual_query}
    Are these queries semantically similar?"""
    
    contents = [
        types.Content(
            role="user", 
            parts=[types.Part.from_text(text=prompt)]
        )
    ]
    
    generate_content_config = types.GenerateContentConfig(
        temperature=0.3,
        top_p=0.95,
        max_output_tokens=8192,
        response_modalities=["TEXT"],
        safety_settings=[
            types.SafetySetting(category="HARM_CATEGORY_HATE_SPEECH", threshold="OFF"),
            types.SafetySetting(category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="OFF"), 
            types.SafetySetting(category="HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold="OFF"),
            types.SafetySetting(category="HARM_CATEGORY_HARASSMENT", threshold="OFF")
        ],
        system_instruction=[types.Part.from_text(text=system_prompt)]
    )
    
    text = ''
    for chunk in client.models.generate_content_stream(
        model="gemini-2.0-flash-001",
        contents=contents,
        config=generate_content_config
    ):
        text += chunk.text
        
    text = text.strip().lower()
    return text == '1'

def evaluate_response(expected_response, actual_response):
    # Evaluate semantic similarity between expected and actual responses
    client = genai.Client(
        api_key="AIzaSyAjXZ9SSdtBshqtgqfup0Mn63R-uto9Q18"
    )
    
    system_prompt = """You are an expert at analyzing semantic similarity between response.
    Compare the following responses and determine if they have the same meaning/intent.
    Respond only with '1' if similar or '0' if not."""
    
    prompt = f"""Query 1: {expected_response}
    Query 2: {actual_response}
    Are these responses semantically similar?"""
    
    contents = [
        types.Content(
            role="user", 
            parts=[types.Part.from_text(text=prompt)]
        )
    ]
    
    generate_content_config = types.GenerateContentConfig(
        temperature=0.3,
        top_p=0.95,
        max_output_tokens=8192,
        response_modalities=["TEXT"],
        safety_settings=[
            types.SafetySetting(category="HARM_CATEGORY_HATE_SPEECH", threshold="OFF"),
            types.SafetySetting(category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="OFF"), 
            types.SafetySetting(category="HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold="OFF"),
            types.SafetySetting(category="HARM_CATEGORY_HARASSMENT", threshold="OFF")
        ],
        system_instruction=[types.Part.from_text(text=system_prompt)]
    )
    
    text = ''
    for chunk in client.models.generate_content_stream(
        model="gemini-2.0-flash-001",
        contents=contents,
        config=generate_content_config
    ):
        text += chunk.text
        
    text = text.strip().lower()
    return text == '0'