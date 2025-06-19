import os
from agent import Agent
from relevant_document import RelevantDocument
from repository import Repository
from dotenv import load_dotenv

# -----------------------------
# Carregar variáveis de ambiente
# -----------------------------
load_dotenv()
os.environ['TAVILY_API_KEY'] = os.getenv("TAVILY_API_KEY")

question = "Chain of Hindsight"

repository = Repository()
retriever = repository.add()

relevantDocument = RelevantDocument()
docs,retrieval_grader = relevantDocument.relevant(question, retriever)
generation,rag_chain = relevantDocument.generate(docs, question)
question_rewriter = relevantDocument.question_rewriter()

agent = Agent(retriever, rag_chain, retrieval_grader, question_rewriter)
app = agent.create_workflow()
agent.run(app)

