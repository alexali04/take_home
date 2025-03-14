from dataclasses import dataclass
from typing import Optional

@dataclass
class Regulatory_API_Prompt():
    """
    Defines API call for regulatory clause extraction.  

    Args:
        (str) model: model to use for API call
        (str) sys_prompt: system prompt for API call
        (str) user_prompt: user prompt for API call
        (int) max_tokens: max tokens for API call
        (str) assistant_prompt: assistant prompt for API call - what the model begins it's response with
    """
    model: str = "claude-3-5-haiku-20241022"
    sys_prompt: str = "You are a regulatory compliance expert. You are given a document and your job is to extract a numbered list of regulatory clauses from the document."
    content: str = "Document: "
    max_tokens: Optional[int] = None
    assistant_prompt: Optional[str] = None