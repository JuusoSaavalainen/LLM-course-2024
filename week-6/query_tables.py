from llmsherpa.readers import LayoutPDFReader
from IPython.core.display import display, HTML
from llama_index.llms.ollama import Ollama
from llama_index.core import VectorStoreIndex
from llama_index.core import Document, ServiceContext, Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings

# Source: https://medium.com/@jitsins/query-complex-pdfs-in-natural-language-with-llmsherpa-ollama-llama3-8b-13b4782243de
# To install:
# 1. run https://stackoverflow.com/questions/52805115/certificate-verify-failed-unable-to-get-local-issuer-certificate
# 2. install and run ollama:
# ollama pull llama3
# ollama run llama3
# 3. Install docker and run:
# docker pull ghcr.io/nlmatics/nlm-ingestor:latest
# docker run -p 5010:5001 ghcr.io/nlmatics/nlm-ingestor:latest
# This will expose the api link “http://localhost:5010/api/parseDocument?renderFormat=all” for you to utilize in your code.

# Initialize LLm
llm = Ollama(model="llama3", request_timeout=2000.0)

llmsherpa_api_url = "http://localhost:5010/api/parseDocument?renderFormat=all"
pdf_url = "https://abc.xyz/assets/91/b3/3f9213d14ce3ae27e1038e01a0e0/2024q1-alphabet-earnings-release-pdf.pdf"
pdf_reader = LayoutPDFReader(llmsherpa_api_url)

# Read PDF
doc = pdf_reader.read_pdf(pdf_url)

# Load all sections into a dictionary
sections = {section.title: section for section in doc.sections()}

print(sections.keys())

# Combine all sections into a single context
combined_context = "\n".join([section.to_html(
    include_children=True, recurse=True) for section in sections.values()])

# save the combined text to a file
# with open("combined_text.txt", "w") as text_file:
#    text_file.write(combined_context)


def ask_question(question):
    resp = llm.complete(
        f"Using the following document context, answer the question:\n{combined_context}\nQuestion: {question}")
    return resp.text


question1 = "What was Google's operating margin for 2024?"
question2 = "What % Net income is of the Revenues?"

# print(ask_question(question1))
# print(ask_question(question2))

# Test reasoning with table data
calculation_question = "Sum up the total revenues for Q1 2024."
print(ask_question(calculation_question))
