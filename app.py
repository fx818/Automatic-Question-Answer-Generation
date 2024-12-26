import streamlit as st
from quesans import helper_tool

st.title("Question Generation from Paragraph")
st.write("This app generates questions from a given paragraph using the T5 model.")

input_paragraph = st.text_input("Enter the paragraph:")

no_of_question = st.number_input("How many questions you want to generate?", min_value=1, max_value=10, value=3)

if input_paragraph != "":
    result = helper_tool(text=input_paragraph, no_of_ques=no_of_question)
    for i, (q, a) in enumerate(result):
        st.write(f"Ques{i+1}: {q}")
        st.write(f"Ans{i+1}: {a}")

