from langchain.llms import HuggingFaceEndpoint
from langchain.prompts import PromptTemplate

# Initialize Hugging Face model
llm = HuggingFaceEndpoint(
    repo_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    task="text-generation"
)

# 1st prompt -> detailed report
template1 = PromptTemplate(
    template='write a detailed report on {topic}',
    input_variables=['topic']
)

# 2nd prompt -> summary
template2 = PromptTemplate(
    template="write a 5-line summary on the following text:\n{text}",
    input_variables=['text']
)

# Invoke first prompt
prompt1 = template1.format(topic="black hole")
result = llm.invoke(prompt1)

# Invoke second prompt
prompt2 = template2.format(text=result)
result2 = llm.invoke(prompt2)

# Print results
print(result)
print("\nSummary:\n", result2)
