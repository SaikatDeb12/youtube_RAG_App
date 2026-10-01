from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

# model = ChatOpenAI(model="gpt-6-luna")
llm = HuggingFaceEndpoint(model="google/gemma-4-31B-it")
model = ChatHuggingFace(llm=llm)

parser = StrOutputParser()
prompt = PromptTemplate(
    template="""
        You are a helpful assistant.

        Answer only using the provided context

        If the context does not contain enough information to answer the question, then just say: "I don't know."

        context:
        {context}

        question:
        {question}
    """,
    input_variables=["context", "question"],
)
# print(prompt)


def generateResult(context: str, query: str) -> str:
    chain = prompt | model | parser
    return chain.invoke({"context": context, "question": query})
