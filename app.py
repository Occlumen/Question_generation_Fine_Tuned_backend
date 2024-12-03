from flask import Flask, jsonify, request
from flask_cors import CORS
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import PromptTemplate
import json
import re
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)

hf_token = os.environ.get('hf_api_token')

# Enable CORS for all routes
CORS(app, resources={r"/*": {"origins": "*"}})

def craete_vector_store(path):
    Loader = PyPDFLoader(path)
    docs = Loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size = 500, chunk_overlap = 100)
    chunks = text_splitter.split_documents(docs[:5])
    embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L12-v2")  
    vector_store = Chroma.from_documents(chunks, embedding)
    return vector_store


def update_json(topic_data, question_total_respose):
    with open('topics.json', 'w') as json_file:
            json.dump(topic_data, json_file, indent=4)  # topic_data is already a dictionary

    with open('questions.json', 'w') as question_json_file:
        json.dump(question_total_respose, question_json_file, indent=4)  # topic_data is already a dictionary

def fetch_ranked_topics(data):
    """
    Parses and fetches the main topics from the input data, which can be either a JSON string or raw text.
    
    Args:
        data (str or dict): The input data containing "main_topics".
    
    Returns:
        dict: A dictionary containing a list of "main_topics".
    """
    main_topics = []

    if isinstance(data, dict):
        # If data is already a dictionary
        main_topics = data.get("main_topics", [])
    elif isinstance(data, str):
        try:
            # Attempt to parse as JSON
            data_dict = json.loads(data)
            main_topics = data_dict.get("main_topics", [])
        except json.JSONDecodeError:
            # Fallback: Use regex to extract main_topics
            match = re.search(r'"main_topics"\s*:\s*\[(.*?)\]', data, re.DOTALL)
            if match:
                topics_content = match.group(1)
                # Clean and split the topics
                main_topics = [topic.strip().strip('"') for topic in topics_content.split(",")]
    
    # Ensure topics are unique and properly cleaned
    main_topics = list(set(main_topics))  # Remove duplicates if needed
    return {"main_topics": main_topics}
    


def extract_mcq_questions(data):
    """Extract MCQ questions from the data."""
    mcq_pattern = re.compile(r'"mcq":\s*({.*?})', re.DOTALL)
    mcq_matches = mcq_pattern.findall(data)
    mcq_questions = []

    for match in mcq_matches:
        try:
            mcq_questions.append(json.loads(match))
        except json.JSONDecodeError:
            pass

    return mcq_questions

def extract_open_ended_questions(data):
    """Extract open-ended questions from the data."""
    open_ended_pattern = re.compile(r'"open_ended":\s*({.*?})', re.DOTALL)
    open_ended_matches = open_ended_pattern.findall(data)
    open_ended_questions = []

    for match in open_ended_matches:
        try:
            open_ended_questions.append(json.loads(match))
        except json.JSONDecodeError:
            pass

    return open_ended_questions

def fetch_ranked_questions(questions):
    """Fetch MCQ and open-ended questions."""
    return {
        "mcq": extract_mcq_questions(questions),
        "open_ended": extract_open_ended_questions(questions)
    }
    

@app.route("/question_generation", methods=['POST'])
def question_generation(): 
    try:
        file = request.files['file'] 
        if file.content_type == 'application/pdf':
            temp_path = os.path.join("temp", file.filename)  # Ensure 'temp' directory exists
            print(temp_path)
            file.save(temp_path)
            vector_store = craete_vector_store(temp_path)
            # Path to the directory containing your .gguf model
            # Use a LLaMA model
            repo_id="mistralai/Mistral-7B-Instruct-v0.2"
            llm=HuggingFaceEndpoint(repo_id=repo_id,max_length=128,temperature=0.2,token=hf_token)
            question = "Give me important and crucial main topics to fully and properly understand "+ str(file.filename)
            print("similarity_search")
            results = vector_store.similarity_search(question, k=5)
            print("similarity_search done")
            content = " ".join(result.page_content for result in results)
            template = """
            Generate five Most important topics or concepts from the data given below :
            {content}
            The output shoul be in json format like the example below:
                "book_title": "Project Management Professional Guide",
                "total_topics": 5,
                "extraction_timestamp": "2024-11-25T10:00:00Z",
                "main_topics": [
                    "Project Initiation",
                    "Project Planning",
                    "Project Execution",
                    "Project Monitoring and Control",
                    "Project Closure"
                ]
            """
            prompt = PromptTemplate(template = template, input_variables = ['content'])
            topic_chain = prompt | llm
            print("Invoke")
            topics = topic_chain.invoke({"content": content})
            topics_dict = fetch_ranked_topics(topics)
            # Access the 'main_topics' field
            main_topics = topics_dict.get("main_topics")
            question_total_respose = {}
            for topic in main_topics:
                topic_content = topic
                ques = "Give me all the important , crucial , valuable and study worthy text about "+ topic
                res = vector_store.similarity_search(ques, k=1)
                question_content = " ".join(rep.page_content for rep in res)
                question_template = """
                    For the following topic {topic_content}. Generate 5 MCQs and 5 open-ended questions with answers using following content:
                    {question_content}
                    Format of  the response should be as follows:
                    "questions": 
                        "mcq": [
                                "topic": "Project Initiation",
                                "question": "What is the primary purpose of a project charter?",
                                "options": ["A. Define schedule", "B. Authorize project", "C. Allocate budget", "D. Identify risks"],
                                "correct_answer": "B",
                                "explanation": "A project charter formally authorizes the project."
                        ],
                        "open_ended": [
                                "topic": "Project Initiation",
                                "question": "Explain the role of a project charter in project success.",
                                "answer": "The project charter establishes scope, objectives, and authority, ensuring project alignment and support."
                        ]
                """
                question_prompt = PromptTemplate(template = question_template, input_variables = ['topic_content', 'question_content'])
                question_chain = question_prompt | llm
                question_response = question_chain.invoke({'topic_content': topic_content, 'question_content': question_content})
                print(question_response)
                question_response_data = fetch_ranked_questions(question_response)
                question_total_respose[topic_content] = question_response_data
            
            print(question_total_respose)
            #topic_data = json.loads(topics)
            update_json(topics_dict, question_total_respose)
            os.remove(temp_path)
            return jsonify({"message": "Question Generation is running", "topic_json": topics_dict, "question_json": question_total_respose}), 200
        else:
            return jsonify({"message": "File is not PDF"}), 200
    except Exception as e:
        print(e)
        return jsonify({"message": "Question Generation is not running", "topic_json": {"Nothing returned ": "Nothing to display"}, "question_json": {"Nothing returned ": "Nothing to display"}}), 200    

@app.route("/", methods=['GET'])
def health_check():
    return jsonify({"message": "API is running"}), 200

if __name__ == '__main__':
    app.run(debug=True)

if __name__ == '__main__':
    app.run(debug=True)
