import streamlit as st
from quesans import helper_tool

st.title("Question Generation from Paragraph")
st.write("This app generates questions from a given paragraph using the T5 model.")

sample_text = ("The symptoms of COVID19 are variable but often include fever, fatigue, cough, "
        "breathing difficulties, loss of smell, and loss of taste. Symptoms may begin one "
        "to fourteen days after exposure to the virus. At least a third of people who are "
        "infected do not develop noticeable symptoms. Of those who develop symptoms "
        "noticeable enough to be classified as patients, most (81%) develop mild to moderate "
        "symptoms (up to mild pneumonia), while 14 develop severe symptoms (dyspnea, hypoxia, "
        "or more than 50 lung involvement on imaging), and 5 develop critical symptoms "
        "(respiratory failure, shock, or multiorgan dysfunction). Older people are at a higher "
        "risk of developing severe symptoms. Some complications result in death. Some people "
        "continue to experience a range of effects (long COVID) for months or years after "
        "infection, and damage to organs has been observed. Multi-year studies are underway "
        "to further investigate the long-term effects of the disease.")



input_paragraph = st.text_input("Enter the paragraph:")

no_of_question = st.number_input("How many questions you want to generate?", min_value=1, max_value=10, value=3)

if input_paragraph != "":
    result = helper_tool(text=input_paragraph, no_of_ques=no_of_question)
    for i, (q, a) in enumerate(result):
        st.write(f"Ques{i+1}: {q}")
        st.write(f"Ans{i+1}: {a}")

