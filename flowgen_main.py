import streamlit as st
from PIL import Image
from dotenv import load_dotenv
import os
load_dotenv()
from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI
from langchain.chains import LLMChain
import subprocess
 
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]

# prompt_template = "generate dedicated data flow diagram for {product}."
prompt_template = '''"Generate a detailed data flow diagram for {product} showcasing all key components and the flow of data between them.

The diagram should:

Include nodes for each major component/process in {product}
Add edges between nodes to visually represent the data flow
Label each node and edge with descriptive names
Organize nodes logically based on the actual workflow (e.g. left to right or top down)
Style nodes differently for clarity (colors, shapes, etc.)
Set graph and node attributes like font, rankdir, etc for readability
Use comments and documentation where helpful to explain the logic
Output the diagram as PNG/SVG image file(s) for easy usage
For reference, {product} is a [describe product and architecture]. The main components are [list key components of product].

Focus on creating a clean, readable diagram that clearly conveys the core data flow for {product}."

'''
 
st.title("Enter your workflow for dfd")
user_input = st.text_input("enter mind chart")
if user_input:
    # user_input = "make dfd for langchain question answring with pdf."
    llm = OpenAI(temperature=0.3)
    llm_chain = LLMChain(
        llm=llm,
        prompt=PromptTemplate.from_template(prompt_template)
    )
    output = llm_chain(user_input)
    code = output['text']
    # st.write("generate dfd",code)

    prompt = '''Understand the flow of this dfd : {code} and Generate full Python code for that dfd.
   
    -give me only python code 
    - add all library like import streamlit as st ...
    -use pydot library for making graph
    -add low line from PIL import Image,from io import BytesIO
    -in last add save graph code graph.write_png(img/data_flow_diagram.png)
    - final add three line to that code "  1st line: image_bytes = graph.create_png(),2nd line : image = Image.open(BytesIO(image_bytes)),3rd line :st.image(image, caption='Data Flow Diagram', use_column_width=True)"
    '''
    llm = OpenAI(temperature=0, openai_api_key = OPENAI_API_KEY,max_tokens=832)

    prompt = f"Generate Python code {prompt}" 

    llm_chain = LLMChain(
        llm=llm,
        prompt=PromptTemplate.from_template(prompt)
    
    )

    output = llm_chain(code)
    st.write("generated the output...")
    code = output['text']

    # st.write(code)

    file_name = "code.py"
    with open(file_name, "w") as file:
        file.write(code)

    # Read the contents of the code.py file
    # with open("code.py", "r") as file:
    #     code_contents = file.read()
    #     st.code(code_contents, language="python")
    #     try:
    #         exec(code_contents)
    #         image = Image.open('data_flow_diagram.png')
    #         if image:
    #             print("image",image)
    #         st.title("Data Flow Diagram :")
    #         st.image(image, caption='Data Flow Diagram', use_column_width=True)
    #     except Exception as e:
    #         st.error(f"Error: {e}")

    with open("code.py", "r") as file:
        code_contents = file.read()
        st.code(code_contents, language="python")
        try:
            st.write("Executing...")
            exec(code_contents)
        except Exception as e:
            st.error(f"Gen Error: {e}")
