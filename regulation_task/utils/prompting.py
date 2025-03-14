from dataclasses import dataclass
from typing import Optional

import anthropic
import dotenv

class Regulatory_API_Prompt:
    """
    Defines API call for regulatory clause extraction.  

    Args:
        (str) model: model to use for API call
        (str) sys_prompt: system prompt for API call, e.g. "You are a reg compliance expert"
        (str) user_prompt: user prompt for API call
        (int) max_tokens: max tokens for API call
        (str) assistant_prompt: assistant prompt for API call - what the model begins it's response with
    """
    def __init__(   
        self,
        model: str = "claude-3-5-haiku-20241022", 
        sys_prompt: str = "You are a regulatory compliance expert. You are given a document and your job is to extract a numbered list of regulatory clauses from the document.", 
        content: str = "Document: ", 
        max_tokens: Optional[int] = None, 
        assistant_prompt: Optional[str] = None
    ):
        
        self.model = model
        self.sys_prompt = sys_prompt
        self.original_content = content
        self.content = content
        self.max_tokens = max_tokens
        self.assistant_prompt = assistant_prompt
    
    def reset_content(self):
        self.content = self.original_content


def extract_clauses_from_docx(api_prompt: Regulatory_API_Prompt, context: str = ""):
    """
    Args:
        (Regulatory_API_Prompt) api_prompt: API prompt to use for clause extraction
        (str) context: context to use for clause extraction
    
    Returns:
        (str) text of clauses
    """

    api_prompt.content += context

    api_key = dotenv.get_key(".env", "ANTHROPIC_API_KEY")

    messages = [
        {"role": "user", "content": api_prompt.content},
    ]

    if api_prompt.assistant_prompt is not None:
        messages.append({"role": "assistant", "content": api_prompt.assistant_prompt})

    if api_prompt.max_tokens is None:
        api_prompt.max_tokens = (len(api_prompt.content) // 4) * 3
    

    client = anthropic.Anthropic(api_key=api_key)
    response = client.messages.create(
        model=api_prompt.model,
        max_tokens=api_prompt.max_tokens,
        system=api_prompt.sys_prompt,
        messages=messages
    )

    return response.content[0].text