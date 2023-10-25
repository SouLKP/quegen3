import streamlit as st  
import tempfile
import pickle
from dotenv import load_dotenv
import os
import openai
import pickle
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
import streamlit as st
load_dotenv()

MODEL="gpt-3.5-turbo"
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
# OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

llm = OpenAI(temperature=0)
memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)

    


def generate_questions(input_dict):
    print(input_dict,'\n **********************') 
    file_path = 'pkl/deepbook.pkl'
    with open(file_path, 'rb') as file:
        book_content_vectorstore = pickle.load(file)

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
    llm = ChatOpenAI(model_name=MODEL, temperature=0.4, openai_api_key='')
    chain = RetrievalQA.from_llm(
            llm=llm,
            retriever=book_content_vectorstore.as_retriever(),
            memory=memory, 
        )

    question_format_instructions = chain(prompt_template)
    # print("instruction :  ",question_format_instructions['result'])
    quepep = question_format_instructions['result']
    return quepep

if 'my_dict' not in st.session_state:
    st.session_state.my_dict = {}


with st.sidebar:
    l1 = []
    st.title("Sectional Details")
    uploaded_file = st.file_uploader("Upload Your book", type="pdf",accept_multiple_files=True)
    pages = []
    for i in uploaded_file:
        mi = uploaded_file[0].file_id
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_file.write(i.getvalue())
            tmp_file_path = tmp_file.name
            loader = PyPDFLoader(tmp_file_path)
            pages += loader.load_and_split()
            print("  Total Pages :",len(pages),"")
    if pages:
        with st.spinner(text="In progress..."):
            embeddings = OpenAIEmbeddings()
            book_content_vectorstore = FAISS.from_documents(pages, embeddings)
            file  = f'pkl/{mi}.pkl'
            if not os.path.exists(file):
                files  = os.listdir('pkl')
                for i in files:
                    os.remove(f'pkl/{i}')
                with open(file, 'wb') as f:
                    pickle.dump(book_content_vectorstore, f)

    with st.form("Sectional Details",clear_on_submit=True,key='kp'):
        Section = st.radio("Select Section Name", ["A", "B", "C"],horizontal = True,index=0,key='section')
        show_A = False
        show_B = False 
        show_C = False
        if Section == "A":
            show_A = True
        elif Section == "B": 
            show_B = True
        else:
            show_C = True
        # print(Section)
        question_type = st.selectbox("Select Question Type", ["MCQS","True / False","Coding Question","Theoritical","Short Question"],placeholder="Ex : mcqs,coding question",index=1)
        question_level = st.selectbox("Select Question Level", ["Easy", "Medium", "Hard"],index=1)
        num_questions = st.number_input("How many questions?", min_value=1,max_value=12, step=1) 
        # Topics = st.text_input("topics",placeholder="ex : machine learning,deep learning")
        countries = ["machine learning", "python", "Computer Vision", "CNN", "AI", "deep learning"]
        Topics = st.multiselect("Choose Topic", countries, ["python"])
        submitted = st.form_submit_button("Submit")
        if submitted:
        #     quedi["option"] = option
        #     quedi["question_type"] = question_type
        #     quedi["question_level"] = question_level
        #     quedi["num_questions"] = num_questions
        #     quedi["num_questions"] = num_questions
            l1.append(Section)
            l1.append(question_type)
            l1.append(question_level)
            l1.append(num_questions)
            l1.append(Topics)
            st.session_state.my_dict[Section] = l1
        reset = st.form_submit_button("reset")


if st.session_state.my_dict:
    with st.expander("See Selected Section details"):
        columns = st.columns(len(st.session_state.my_dict))

        # A list to collect keys to be removed
        keys_to_remove = []
        
        
        # Display data and add "Remove" button for each column 
        for col, (key, value) in zip(columns, st.session_state.my_dict.items()):
            col.subheader(f"Section {key}")
            col.write(value)
            # Use a "Remove" button to add the key to keys_to_remove list
            if col.button(f"Double-Click to Remove {key}"):
                keys_to_remove.append(key)

        # Remove the selected keys
        for key in keys_to_remove:
            del st.session_state.my_dict[key]

if st.session_state.my_dict:
    op = st.button("Generate question paper",type='primary',key='dif2')
    if op: 
        input_dict = st.session_state.my_dict
        question_paper_result = generate_questions(input_dict)
        st.write("Generate Question Paper")
        st.write(question_paper_result)
        if st.download_button("Download as Text File",data=question_paper_result,type='primary',key='non'): 
            st.success("Text file downloaded!")  
            
    
