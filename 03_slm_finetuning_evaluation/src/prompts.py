def build_eval_prompt(problem: str) -> str:
    return f'''You are an SDG community assistant.

Given the local community problem, return:
Problem Category:
Relevant SDGs:
Suggested Actions:
Safety Note:
Limitations:

Problem:
{problem}
'''
