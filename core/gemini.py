import google.genai as genai
from typing import List, Optional, Any

class Gemini:
    def __init__(self, model: str, api_key: str):
        self.model = model
        self.api_key = api_key
        # The client is initialized here.
        self.client = genai.Client(api_key=self.api_key)

    def add_user_message(self, messages: list, message: Any):
        content = message.content if isinstance(message, genai.types.GenerateContentResponse) else message
        user_message = {"role": "user", "parts": content}
        messages.append(user_message)

    def add_assistant_message(self, messages: list, message: Any):
        assistant_message = {"role": "model", "parts": message.parts}
        messages.append(assistant_message)

    def text_from_message(self, message: genai.types.GenerateContentResponse) -> str:
        text_parts = [
            part.text
            for part in message.parts
            if isinstance(part, str) and not part.function_call
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
    ) -> genai.types.GenerateContentResponse:
        params = {
            "model": self.model,
            "contents": messages,
            "temperature": temperature,
        }

        if tool_config:
            params["tool_config"] = {"tools": tool_config}

        # System instructions are handled differently in Gemini
        # by prepending a system message to the messages list.
        if system:
            system_message = {"role": "user", "parts": [system]}
            params["contents"].insert(0, system_message)
            
        # Call the Gemini API
        response = self.client.models.generate_content(
            model=params["model"],
            contents=params["contents"],
            tool_config=params.get("tool_config"),
            generation_config=genai.types.GenerationConfig(
                temperature=params["temperature"],
            )
        )
        return response