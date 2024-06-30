#Load OpenAI API Key
#os.environ["OPENAPI_API_KEY"] = openai.api_key
from openai import OpenAI
#from openai.embeddings_utils import get_embedding
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
#from langchain.embeddings.openai import OpenAIEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI


#Setup Loader - in this case a PDF Loader
loader = PyMuPDFLoader("suse_administration.pdf")

#Load and split the pdf into pages
pages = loader.load_and_split()

# setup a text splitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100,
    length_function=len,
)

#split pages into smaller chunks called docs
docs = text_splitter.split_documents(pages)

#transform to embeddings
embeddings = OpenAIEmbeddings()

#setup and store docs and embeddings into ChromaDB
vectordb = Chroma.from_documents(docs, embedding=embeddings,
                                 persist_directory=".")

#Make the database persisten
vectordb.persist()

#setup memory so it remembers previous questions and answers
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

#Perform the conversational Retrieval Chain
qa = ConversationalRetrievalChain.from_llm(ChatOpenAI(temperature=0.5),vectordb.as_retriever(), memory=memory)

#Run the question
question = "Explain grouping and combining commands in Suse linux.?"
result = qa.run(question)

#print the values to the screen
print(result)
