from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate

template = (
    "Your task is to extract specific information from the following text: {dom_content}.\n\n"
    "Please follow these instructions precisely:\n\n"
    "1. **Match Exact Phrases:** Extract only the information that directly matches the exact wording or phrases in "
    "the description: {parse_description}.\n"
    "2. **Exclude Non-Matching Content:** Do not include any text that does not match the description exactly.\n"
    "3. **Return an Empty String if No Match:** If no exact match is found, return an empty string ('').\n"
    "4. **Output Only the Matched Data:** Provide only the extracted data, without any additional text or explanation."
)


model = OllamaLLM(model="llama3.1")


def parse_with_ollama(dom_chunks, parse_description):
    prompt = ChatPromptTemplate.from_template(template)
    chain = prompt | model

    parsed_result = []

    for i, chunk in enumerate(dom_chunks, start=1):
        response = chain.invoke(
            {"dom_content": chunk, "parse_description": parse_description}
        )
        print(f"Parsed batch {i} of {len(dom_chunks)}")
        parsed_result.append(response)

        return "\n".join(parsed_result)
