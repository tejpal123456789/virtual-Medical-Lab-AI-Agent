def estimate_tokens(text: str) -> int:
    # Simple heuristic; replace with tiktoken if needed
    return max(1, int(len(text.split()) * 1.3)) 