from langchain_groq import ChatGroq
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_groq import ChatGroq

# Initialize the language model
from flipkart.config import Config
llm = ChatGroq(model=Config.RAG_MODEL, temperature=0.5)

# Create the main prompt template with custom document variable name
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Based on the provided information, answer the question."),
    ("human", "Information: {context}\n\nQuestion: {question}")
])

# Create a custom document prompt for formatting each document
document_prompt = PromptTemplate(
    input_variables=["page_content", "source"],
    template="Source: {source}\nContent: {page_content}"
)

# Create a custom output parser
output_parser = StrOutputParser()

# Create the stuff documents chain with all parameters
chain = create_stuff_documents_chain(
    llm=llm,
    prompt=prompt,
    output_parser=output_parser,
    document_prompt=document_prompt,
    document_separator="\n---\n",  # Custom separator between documents
    document_variable_name="context"  # Custom variable name instead of default "context"
)

# Create sample documents with metadata
docs = [
    Document(
        page_content="Python is a high-level programming language known for its simplicity and readability.",
        metadata={"source": "Programming Guide"}
    ),
    Document(
        page_content="Python supports multiple programming paradigms including procedural, object-oriented, and functional programming.",
        metadata={"source": "Language Features"}
    ),
    Document(
        page_content="Python has a vast ecosystem of libraries and frameworks for various applications.",
        metadata={"source": "Ecosystem Overview"}
    )
]

# Invoke the chain
result = chain.invoke({
    "context": docs,
    "question": "What are the key characteristics of Python?"
})

print(result)