from langchain_huggingface import HuggingFacePipeline

llm = HuggingFacePipeline.from_model_id(
      model_id= "microsoft/phi-3-mini-4k-instruct",
      task = "text-generation",
      pipeline_kwargs={"temperature":0.1,"max_new_tokens":100,
                       "top_k":50},
    )
llm.invoke("Langchain is all about")

#langchain with huggingfaceEnd point
from google.colab import userdata
import os
os.environ["HF_TOKEN"]=userdata.get("HF_TOKEN")

from langchain_huggingface import HuggingFaceEndpoint

llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Meta-llama-3-8B-Instruct",
    task = "text-generation",
    max_new_tokens=100,
    do_sample=False,
)

llm.invoke("Hugging Face is ")
