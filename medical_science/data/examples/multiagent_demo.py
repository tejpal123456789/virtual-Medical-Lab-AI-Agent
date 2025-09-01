from medical_science.src.agents import process

if __name__ == "__main__":
    result = process("Hello, what is COVID-19?", conversation_history=None)
    print(result) 