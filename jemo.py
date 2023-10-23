import os
from dotenv import load_dotenv
load_dotenv()
import pickle
import openai
from langchain.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain.llms import OpenAI
from langchain.chains import VectorDBQA
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.document_loaders import TextLoader
from langchain.vectorstores import FAISS
from langchain.document_loaders import PyPDFLoader
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import ConversationalRetrievalChain, RetrievalQA
from langchain.llms import HuggingFaceHub


# load_dotenv()
# MODEL = os.getenv("MODEL")
# OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

import streamlit as st
# openai.api_key=st.secrets["OPENAI_API_KEY"]
huggingfacehub_api_token = st.secrets["HUGGINGFACEHUB_API_TOKEN"]
MODEL = "gpt-3.5-turbo"

llm = OpenAI(temperature=0)

# Define ConversationBufferMemory
memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)

# pdf_file_path = "/home/webclues/Desktop/deep_learning.pdf"
# pages1 = []
# loader = PyPDFLoader(pdf_file_path)
# pages1 += loader.load_and_split()
# print("Total Pages of Book", len(pages1))

# loader = TextLoader("laste.txt") 
# documents = loader.load()
# text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
# texts = text_splitter.split_documents(documents)

# embeddings = OpenAIEmbeddings()
# book_content_vectorstore = FAISS.from_documents(pages1, embeddings)
 
import pickle

def generate_questions(input_dict):
    # print(input_dict,'\n **********************') 
    file_path = 'pkl/deepbook.pkl'
    # if not os.path.exists(file_path):
    #     with open('pkl/deepbook.pkl', 'wb') as f:
    #         pickle.dump(book_content_vectorstore, f)
    #         print("pkl done")

    with open(file_path, 'rb') as file:
        book_content_vectorstore = pickle.load(file)
        # questions_pattern_vectorestore = FAISS.from_documents(texts, embeddings)

    prompt_template = "Generate questions from the book content with the following section-wise instructions:"
    output_template = ""

    # Iterate through the dynamic dictionary
    for section, values in input_dict.items():
        section_name, question_type, difficulty, num_questions,topic  = values
        prompt_template += f"\n\nSection {section_name}:"
        prompt_template += f"\nQuestion type: {question_type}"
        prompt_template += f"\nDifficulty level: {difficulty}"
        prompt_template += f"\nTotal number of Question : {num_questions}"
        prompt_template += f"\n Topic : {topic}"

        output_template += f"\n\nSection {section_name}: instruction : {question_type}"
        output_template += f"\nQuestion: [Generated {question_type} question]"
        output_template += f"\nAnswer: [Answer option letter]"

    last_message = "\n The questions should demonstrate understanding of key concepts from the book content. Format the full question paper clearly labeling the sections and numbering the questions sequentially."    
    
    prompt_template = prompt_template + output_template + last_message
    # print(prompt_template,"************************")
    # prompt_template = '''
    # Generate a questions from the book content with the following section-wise instructions:

    # Section A:

    # Question type: MCQ
    # Topic: Computer Vision, Machine Learning algorithms
    # Difficulty level: Medium
    # Section B:

    # Question type: Fill in the blank
    # Topic: Deep Learning, CNNs
    # Difficulty level: Medium
    # Section C:

    # Question type: Practical question
    # Topic: Transformers
    # Difficulty level: Medium
    # Section D:

    # Question type: Theoretical question
    # Topic: Generative AI
    # Difficulty level: Medium 
    # Section E:

    # Question type: Graph 
    # Topic: generate graph
    # Difficulty level: Medium
    # Output the questions in the following format:

    # Section A: instruction : Question type
    # Question: [Generated MCQ question]
    # Answer: [Answer option letter]

    # Section B: instruction : Fill in the blank
    # Question: [Generated blank question]
    # Answer: [Fill in the blank answer]

    # And so on for Sections C and D...

    # The questions should demonstrate understanding of key concepts from the book content. Format the full question paper clearly labeling the sections and numbering the questions sequentially.
    # '''
    
    repo_id = "meta-llama/Llama-2-70b-chat-hf"
    llm = HuggingFaceHub(repo_id=repo_id,verbose=False, model_kwargs={
                         "temperature": 0.2, "max_seq_len": 4000, "max_new_tokens": 2048})
    
    
    llm = ChatOpenAI(model_name=MODEL, temperature=0.4)
    chain = RetrievalQA.from_llm(
            llm=llm,
            retriever=book_content_vectorstore.as_retriever(),
            memory=memory, 
        )

    question_format_instructions = chain(prompt_template)
    print("instruction :  ",question_format_instructions['result'])
    quepep = question_format_instructions['result']
    # quepep = "Done"

    # from langchain.agents import load_tools
    # tools = load_tools(['dalle-image-generator'])

    # # Initialize agent
    # from langchain.agents import initialize_agent
    # agent = initialize_agent(tools, llm, agent="zero-shot-react-description", verbose=True)
    
    # prompt = """
    # Generate an exam question about "A diagram explaining pythagoras theorem with some angle values labeled":

    # [DALL-E generated image of "A diagram explaining pythagoras theorem with some angle values labeled"]

    # The question text should ask the student to analyze the image and apply concepts to explain the pythagoras. Leave space for the student to write their explanatory answer after the question.

    # """

    # output = agent.run(prompt)
 
    # print(output)
    print(quepep)
    return quepep

# if __name__ == "__main__":
#     result = generate_questions()
#     print(result)
