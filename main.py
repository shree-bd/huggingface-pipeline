from transformers import pipeline
from langchain_huggingface import HuggingFacePipeline
from langchain_core.prompts import PromptTemplate
import torch

# Create summarization pipeline
summarization_pipeline = pipeline(task="summarization", model="facebook/bart-large-cnn", device="mps")
summarizer = HuggingFacePipeline(pipeline=summarization_pipeline)

# Create refinement pipeline (using a smaller model that works well on M4)
refinement_pipeline = pipeline(task="summarization", model="facebook/bart-large", device="mps")
refiner = HuggingFacePipeline(pipeline=refinement_pipeline)

# Create question-answering pipeline
qa_pipeline = pipeline(task="question-answering", model="deepset/roberta-base-squad2", device="mps")

# Create summary template
summary_template = PromptTemplate.from_template("Summarize the following text in a {length} way:\n\n{text}")

# Create summarization chain
summarization_chain = summary_template | summarizer | refiner

# Get input text from user
text_to_summarize = input("\nEnter text to summarize:\n")

# Get desired length
length = input("\nEnter the length (short/medium/long): ")

# Generate summary
summary = summarization_chain.invoke({"text": text_to_summarize, "length": length})

print("\n🤖 **Generated Summary:**")
print(summary)
