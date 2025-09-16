import os
import google.generativeai as genai
import dotenv

dotenv.load_dotenv()

# Configure your API key. It's recommended to set it as an environment variable
# and then access it like this.
# On your terminal, run: export GEMINI_API_KEY='your_api_key'
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError("API key not found. Please set the GEMINI_API_KEY environment variable.")

genai.configure(api_key=api_key)

generation_config = genai.GenerationConfig(
    temperature=0.9,
    max_output_tokens=200,
)

# Initialize model and pass configuration
model = genai.GenerativeModel(
    'gemini-1.5-flash',
    generation_config=generation_config
)

response = model.generate_content("hi")
print(response.text)