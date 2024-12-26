import spacy
from transformers import pipeline



# import subprocess

# try:
#     subprocess.run(
#         [
#             "python",
#             "-m",
#             "pip",
#             "install",
#             "--user",
#             "https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.5.0/en_core_web_sm-3.5.0.tar.gz",
#         ],
#         check=True,
#     )
#     print("Model installed successfully with --user!")
# except subprocess.CalledProcessError as e:
#     print(f"Error during installation: {e}")






# spacy.cli.download("en_core_web_sm")

# Load SpaCy model for NER
model_path = "en_core_web_sm-3.5.0/en_core_web_sm/en_core_web_sm-3.5.0/"
nlp_spacy = spacy.load(model_path)

# Load the question generation model
nlp_qg = pipeline("text2text-generation", model="valhalla/t5-base-qg-hl")

# The model valhalla/t5-base-qg-hl is a pre-trained model hosted on the Hugging Face Model Hub. It's a variant of the T5 (Text-To-Text Transfer Transformer) model fine-tuned for Question Generation (QG) tasks.

# Load the question answering model
nlp_qa = pipeline("question-answering", model="distilbert-base-cased-distilled-squad")

# The distilbert-base-cased-distilled-squad model is a pre-trained model available on the Hugging Face Model Hub. It is a distilled version of the BERT (Bidirectional Encoder Representations from Transformers) model, specifically fine-tuned for the SQuAD (Stanford Question Answering Dataset) task.

# DistilBERT is a smaller, faster, cheaper, and lighter version of BERT. It is created using a process called knowledge distillation, where a smaller model (the student) is trained to replicate the behavior of a larger model (the teacher), in this case, BERT. DistilBERT retains 97% of BERT's language understanding capabilities while being 60% faster and 40% smaller in size.

# This specific model, distilbert-base-cased-distilled-squad, is a variant of DistilBERT that has been fine-tuned on the SQuAD dataset, which is a large-scale dataset for question answering tasks. 


def highlight_text(text, entity):
    highlighted_text = text.replace(entity, f"<hl>{entity}<hl>")
    return highlighted_text


# Example input text
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


def helper_tool(text, no_of_ques):

    # Use SpaCy to identify named entities in the text
    doc = nlp_spacy(text)
    # entities = [ent.text for ent in doc.ents]
    entities = []
    for ent in doc.ents:
        if ent.text not in entities:
            entities.append(ent.text)
    print(entities)

    # Set the number of questions
    no_of_questions = no_of_ques

    # Generate questions by highlighting different entities
    questions_and_answers = []
    for entity in entities:
        highlighted_text = highlight_text(text, entity)
        result = nlp_qg(highlighted_text, max_new_tokens=50)
        question = result[0]['generated_text']
        # Get the answer for the generated question
        answer = nlp_qa(question=question, context=text)['answer']
        questions_and_answers.append((question, answer))
        if len(questions_and_answers) >= no_of_questions:
            break

    # If not enough questions, generate more variations
    if len(questions_and_answers) < no_of_questions:
        for _ in range(no_of_questions - len(questions_and_answers)):
            highlighted_text = highlight_text(text, entities[0])
            result = nlp_qg(highlighted_text, max_new_tokens=50)
            question = result[0]['generated_text']
            answer = nlp_qa(question=question, context=text)['answer']
            questions_and_answers.append((question, answer))


    # Print the generated questions and answers
    for i, (question, answer) in enumerate(questions_and_answers[:no_of_questions]):
        print(f"Question {i+1}: {question}")
        print(f"Answer {i+1}: {answer}\n")

    return questions_and_answers

# helper_tool(sample_text, 2)