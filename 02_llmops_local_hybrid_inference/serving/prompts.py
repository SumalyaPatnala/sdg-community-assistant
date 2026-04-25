def build_prompt(task: str, text: str) -> str:
    task = task.lower().strip()

    if task == "summarization":
        return f"""
You are a production summarization assistant.

Summarize the input clearly and concisely.

Return:
Summary:
- 
Key Points:
- 
Limitations:
- 

Input:
{text}
"""

    if task == "extraction":
        return f"""
You are an information extraction assistant.

Extract structured fields from the input.
If a field is missing, write "Not mentioned".

Return JSON with:
- entities
- dates
- organizations
- risks
- actions
- metrics

Input:
{text}
"""

    if task == "classification":
        return f"""
You are a classification assistant.

Classify the input into one or more categories.
Explain the reason and confidence.

Return:
Category:
Reason:
Confidence:

Input:
{text}
"""

    return f"""
You are a general LLM inference assistant.

Task:
{task}

Input:
{text}

Return a clear and structured answer.
"""
