from typing import List, Optional, Any
from google.generativeai import GenerativeModel, types

class Gemini:
    # Removed api_key from __init__ as it's configured globally
    def __init__(self, model: str):
        self.model = model
        self.client = GenerativeModel(self.model)

    def add_user_message(self, messages: list, message: Any):
        content = message.text if hasattr(message, 'text') else message
        user_message = {"role": "user", "parts": [content]}
        messages.append(user_message)

    def add_assistant_message(self, messages: list, message: Any):
        assistant_message = {"role": "model", "parts": message.parts}
        messages.append(assistant_message)

    def text_from_message(self, message: types.GenerateContentResponse) -> str:
        text_parts = [
            part.text
            for part in message.parts
            if hasattr(part, 'text') and part.text is not None and not hasattr(part, 'function_call')
        ]
        return "\n".join(text_parts)

    def chat(
        self,
        messages,
        system=None,
        temperature=1.0,
        stop_sequences=None,
        tool_config: Optional[list] = None,
        thinking=False,
        thinking_budget=1024,
    ) -> types.GenerateContentResponse:
        params = {
            "model": self.model,
            "contents": messages,
            "temperature": temperature,
        }

        if tool_config:
            params["tools"] = tool_config

        if system:
            system_message = {"role": "user", "parts": [system]}
            params["contents"].insert(0, system_message)
            
        response = self.client.generate_content(
            contents=params["contents"],
            tools=params.get("tools"),
            generation_config=types.GenerationConfig(
                temperature=params["temperature"],
            )
        )
        return response