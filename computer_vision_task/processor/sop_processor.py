import anthropic
import dotenv
from typing import Optional
import docx

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


def get_discrepancies(api_prompt: Regulatory_API_Prompt, context: str = ""):
    """
    Args:
        (Regulatory_API_Prompt) api_prompt: API prompt to use for clause extraction
        (str) context: context to use for clause extraction
    
    Returns:
        (str) text of clauses
    """

    api_prompt.content += context

    print(f"Context Length: {len(api_prompt.content)}")

    api_key = dotenv.get_key(".env", "ANTHROPIC_API_KEY")

    messages = [
        {"role": "user", "content": api_prompt.content},
    ]

    if api_prompt.assistant_prompt is not None:
        messages.append({"role": "assistant", "content": api_prompt.assistant_prompt})

    if api_prompt.max_tokens is None:
        api_prompt.max_tokens = (len(api_prompt.content) // 4) * 3
    

    client = anthropic.Anthropic(api_key=api_key)

    all_text = []

    with client.messages.stream(
        model=api_prompt.model,
        max_tokens=api_prompt.max_tokens,
        system=api_prompt.sys_prompt,
        messages=messages,
    ) as stream:
        for text in stream.text_stream:
            all_text.append(text)

    return "".join(all_text)


def read_sop(sop_path: str):
    """
    Args:
        (str) sop_path: path to SOP

    Since sop is a docx file, we need to read it as a docx file.
    """
    doc = docx.Document(sop_path)
    paras = []
    for para in doc.paragraphs:
        paras.append(para.text)
    return "\n".join(paras)
