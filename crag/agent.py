from typing import List
from typing_extensions import TypedDict
from langchain.schema import Document
from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.graph import END, StateGraph, START
from pprint import pprint

class GraphState(TypedDict):
    """
    Representa o estado do nosso grafo.

    Atributos:
        question: pergunta
        generation: geração LLM
        web_search: se adicionar busca na web
        documents: lista de documentos
    """

    question: str
    generation: str
    web_search: str
    documents: List[str]

class Agent:
    def __init__(self, retriever, rag_chain, retrieval_grader, question_rewriter):
        self.retriever = retriever
        self.rag_chain = rag_chain
        self.retrieval_grader = retrieval_grader
        self.question_rewriter = question_rewriter
        self.web_search_tool = TavilySearchResults(k=3)

    def recuperar(self,state):
      """
      Recuperar documentos
      """
      print("---RECUPERAR DOCUMENTOS---")
      question = state["question"]
      documents = self.retriever.get_relevant_documents(question)
      return {"documents": documents, "question": question}
    
    def gerar(self, state):
      """
      Geração de resposta
      """
      print("---GERAR---")
      question = state["question"]
      documents = state["documents"]
      generation = self.rag_chain.invoke({"context": documents, "question": question})
      return {"documents": documents, "question": question, "generation": generation}
    
    def avaliar_documentos(self, state):
      """
      Determina se os documentos recuperados são relevantes à pergunta.
      """
      print("---VERIFICAR RELEVÂNCIA DOS DOCUMENTOS À PERGUNTA---")
      question = state["question"]
      documents = state["documents"]

      filtered_docs = []
      web_search = "Não"
      for d in documents:
          score = self.retrieval_grader.invoke(
              {"question": question, "document": d.page_content}
          )
          grade = score.binary_score
          if grade == "sim":
              print("---CONCLUSÃO: DOCUMENTO RELEVANTE---")
              filtered_docs.append(d)
          else:
              print("---CONCLUSÃO: DOCUMENTO NÃO RELEVANTE---")
              web_search = "Sim"
              continue
      return {"documents": filtered_docs, "question": question, "web_search": web_search}
    
    def transformar_pergunta(self,state):
      """
      Transformar a pergunta para produzir uma pergunta melhor.
      """
      print("---REESCREVER PERGUNTA---")
      question = state["question"]
      documents = state["documents"]
      better_question = self.question_rewriter.invoke({"question": question})
      return {"documents": documents, "question": better_question}
    
    def busca_web(self,state):
      """
      Busca na web com base na pergunta reescrita.
      """
      print("---BUSCA NA WEB---")
      question = state["question"]
      documents = state["documents"]
      docs = self.web_search_tool.invoke({"query": question})
      web_results = "\n".join([d["content"] for d in docs])
      web_results = Document(page_content=web_results)
      documents.append(web_results)
      return {"documents": documents, "question": question}
    
    def decidir_geracao(self,state):
      """
      Determina se deve gerar uma resposta ou re-gerar uma pergunta.
      """
      print("---AVALIAR CONCLUSÃO DOS DOCUMENTOS---")
      web_search = state["web_search"]

      if web_search == "Sim":
          print("---DECISÃO: NENHUM DOCUMENTO RELEVANTE À PERGUNTA, TRANSFORMAR PERGUNTA---")
          return "transformar_pergunta"
      else:
          print("---DECISÃO: GERAR---")
          return "gerar"
      
    def create_workflow(self):
      # Criar o grafo

      workflow = StateGraph(GraphState)

      # Definir os nós
      workflow.add_node("recuperar", self.recuperar)
      workflow.add_node("avaliar_documentos", self.avaliar_documentos)
      workflow.add_node("gerar", self.gerar)
      workflow.add_node("transformar_pergunta", self.transformar_pergunta)
      workflow.add_node("busca_web", self.busca_web)

      # Construir grafo
      workflow.add_edge(START, "recuperar")
      workflow.add_edge("recuperar", "avaliar_documentos")
      workflow.add_conditional_edges(
        "avaliar_documentos",
        self.decidir_geracao,
        {
          "transformar_pergunta": "transformar_pergunta",
          "gerar": "gerar",
        },
      )
      workflow.add_edge("transformar_pergunta", "busca_web")
      workflow.add_edge("busca_web", "gerar")
      workflow.add_edge("gerar", END)

      # Compilar
      app = workflow.compile()
      return app
    
    def run(self, app):
        inputs = {"question": "Quais são os tipos de memória de agentes?"}
        for output in app.stream(inputs):
          for key, value in output.items():
            pprint(f"Nó '{key}':")
            pprint("\n---\n")

        # Resposta final
        print("\nResposta final:")
        pprint(value["generation"])