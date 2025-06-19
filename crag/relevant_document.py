from langchain import hub
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

class GradeDocuments(BaseModel):
    """Pontuação binária para verificação de relevância em documentos recuperados."""

    binary_score: str = Field(
        description="Os documentos são relevantes à pergunta, 'sim' ou 'não'"
    )

class RelevantDocument:
    """
    Represents a relevant document in the context of a query.
    Contains the document ID, its content, and the score indicating its relevance.
    """

    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-4.1-nano", temperature=0)
        self.structured_llm_grader = self.llm.with_structured_output(GradeDocuments)
        self.prompt = hub.pull("rlm/rag-prompt")

    def relevant(self,question, retriever):
      system = """Você é um avaliador que verifica a relevância de um documento recuperado em relação a uma pergunta do usuário, e indepentente da língua do documento, todas as respostas a respeito do documento devem ser em português do Brasil. \n 
                  Se o documento contém palavras-chave(s) ou significado semântico relacionado à pergunta, classifique-o como relevante. \n
                  Dê uma pontuação binária 'sim' ou 'não' para indicar se o documento é relevante à pergunta."""
      grade_prompt = ChatPromptTemplate.from_messages(
        [
          ("system", system),
          ("human", "Documento recuperado: \n\n {document} \n\n Pergunta do usuário: {question}"),
        ]
      )
      docs = retriever.get_relevant_documents(question)
      retrieval_grader = grade_prompt | self.structured_llm_grader
      doc_txt = docs[1].page_content
      print(retrieval_grader.invoke({"question": question, "document": doc_txt}))
      #question = "Chain of Hindsight"
      return docs, retrieval_grader
    
    def generate(self, docs, question):
        """
        Generates a response based on the retrieved documents and the user's question.
        """
        rag_chain = self.prompt | self.llm | StrOutputParser()
        generation = rag_chain.invoke({"context": docs, "question": question})
        return generation,rag_chain
    
    def question_rewriter(self):
        """
        Rewrites the user's question to improve retrieval accuracy.
        """
        system = """Você é um reescritor de pergunta que converte uma pergunta de entrada em uma versão melhorada que é otimizada \n 
                    para busca na web. Analise a entrada e tente entender o significado subjacente. Todas as respostas a respeito do documento devem ser em português do Brasil."""
        re_write_prompt = ChatPromptTemplate.from_messages(
          [
            ("system", system),
            (
              "human",
              "Aqui está a pergunta inicial: \n\n {question} \n Formule uma pergunta melhorada.",
            ),
          ]
        )
        question_rewriter = re_write_prompt | self.llm | StrOutputParser()
        return question_rewriter