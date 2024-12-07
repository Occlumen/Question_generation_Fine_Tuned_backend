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
    

def extract_questions_data(text):
    """
    Extracts JSON-like data starting from 'questions:' in the provided text.

    Args:
    text (str): The input text containing JSON-like content.

    Returns:
    str: Extracted JSON-like data starting from 'questions:', or a message indicating no match found.
    """
    # Regex pattern to match everything after "questions":
    pattern = re.compile(r"\"questions\":\s*{.*", re.DOTALL)
    match = pattern.search(text)
    
    if match:
        # Extract and return the matched data
        return match.group(0).strip()
    else:
        return "No questions data found."
    

@app.route("/question_generation", methods=['POST'])
def question_generation(): 
    try:
        file = request.files['file'] 
        if file.content_type == 'application/pdf':
            temp_path = os.path.join("temp", file.filename)  # Ensure 'temp' directory exists
            print(temp_path)
            file.save(temp_path)
            vector_store = craete_vector_store(temp_path)
            print("Vector Store Created")
            # Path to the directory containing your .gguf model
            # Use a LLaMA model
            repo_id="mistralai/Mistral-7B-Instruct-v0.2"
            llm=HuggingFaceEndpoint(repo_id=repo_id,max_length=128,temperature=0.2,token=hf_token)
            question = "Give me important and crucial main topics to fully and properly understand "+ str(file.filename)
            print("similarity_search")
            results = vector_store.similarity_search(question, k=2)
            print("similarity_search done")
            content = " ".join(result.page_content for result in results)
            topic_examples = """
            The output should be foloowing the below format:
            Example 1 (JSON):
            {
            "main_topics": [
                    "Project Initiation",
                    "Project Planning",
                    "Project Execution",
                    "Project Monitoring and Control",
                    "Project Closure"
                ]
                }
            Example 2 (JSON):
            {
            "main_topics": [
                    "Journalism ethics",
                    "Investigative journalism",
                    "Online journalism",
                    "Local journalism",
                    "Citizen journalism"
                ]
                }
            Example 3 (JSON):
            {
            "main_topics": [
                    "Social engineering",
                    "Data privacy",
                    "Internet of things (IoT)",
                    "Network security",
                    "Password management"
                ]
                }
            Example 4 (JSON):
            {
            "main_topics": [
                    "Data visualization",
                    "Machine learning",
                    "Deep learning",
                    "Natural language processing (NLP)",
                    "Data mining"
                ]
                }
            """
            template = """
            You are greater reader and can understand and comprehend anything. Read the context below to get a better understanding and
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
            Following are the examples you can refer to:
            {topic_examples}
            """
            prompt = PromptTemplate(template = template, input_variables = ['content'])
            topic_chain = prompt | llm
            print("Invoke")
            topics = topic_chain.invoke({"content": content, "topic_examples": topic_examples})
            print("Topics Generated")
            topics_dict = fetch_ranked_topics(topics)
            # Access the 'main_topics' field
            main_topics = topics_dict.get("main_topics")
            question_total_respose = {}
            for topic in main_topics:
                topic_content = topic
                ques = "Give me all the important , crucial , valuable and study worthy text about "+ topic
                res = vector_store.similarity_search(ques, k=1)
                question_content = " ".join(rep.page_content for rep in res)
                question_examples = """
                    The output should be following the below format:
                    Example 1 (JSON):
                    "questions": {
                        "mcq": [
                            {
                                "topic": "Cell Biology",
                                "type": "MCQ",
                                "question": "In which of the following type of cells the cell junction is abundant?",
                                "options": [
                                    "A) Cardiac cells",
                                    "B) Prokaryotic cells",
                                    "C) Hepatic cells",
                                    "D) Epithelial cells"
                                ],
                                "correct_answer": "D",
                                "explanation": "The cell junction is abundant in epithelial cells, which provide barrier and control over the transport in the cell. It is otherwise known as intercellular bridge, which is made up of multiprotein complexes."
                            }
                        ],
                        "open_ended": [
                            {
                                "topic": "Cell Biology",
                                "type": "open_ended",
                                "question": "What do you mean by plasmids? What role do they play in bacteria ?",
                                "model_answer": "A plasmid is an autonomously replicating, extra-chromosomal ....  remain separate from the chromosome.",
                                "key_points": [
                                    "double-stranded DNA",
                                    "recombination experiments",
                                    "bacterial conjugation"
                                ]
                            }
                        ]
                    }
                    Example 2 (JSON):
                    "questions": {
                        "mcq": [
                            {
                                "topic": "Data Structure",
                                "type": "MCQ",
                                "question": "What are the disadvantages of arrays",
                                "options": [
                                    "A) Index value of an array can be negative",
                                    "B) Elements are sequentially accessed",
                                    "C) Data structure like queue or stack cannot be implemented",
                                    "D) There are chances of wastage of memory space if elements inserted in an array are lesser than the allocated size"
                                ],
                                "correct_answer": "D",
                                "explanation": "Arrays are of fixed size. If we insert elements less than the allocated size, unoccupied positions can’t be used again. Wastage will occur in memory."
                            }
                        ],
                        "open_ended": [
                            {
                                "topic": "Data Structure",
                                "type": "open_ended",
                                "question": "What is a linked list Data Structure ?",
                                "model_answer": "This is one of the most frequently asked data structure interview questions where the ...... and the list has the ability to grow and shrink on demand.",
                                "key_points": [
                                    "dynamic data structure",
                                    "djacent memory locations",
                                    "pointers to form a chain"
                                ]
                            }
                        ]
                    }
                    Example 3 (JSON):
                    "questions": {
                        "mcq": [
                            {
                                "topic": "Economics",
                                "type": "MCQ",
                                "question": "Which of the following is the relation that the law of demand defines?",
                                "options": [
                                    "A) Income and price of a commodity",
                                    "B) Price and quantity of a commodity",
                                    "C) Income and quantity demanded",
                                    "D) Quantity demanded and quantity supplied"
                                ],
                                "correct_answer": "B",
                                "explanation": "..."
                            }
                        ],
                        "open_ended": [
                            {
                                "topic": "Economics",
                                "type": "open_ended",
                                "question": "Outline the difference between positive and normative questions?",
                                "model_answer": "Positive economics focuses on the description, quantification, and explanation of economic developments, expectations, and associated phenomena. It relies on objective data analysis, fact-based (precise, descriptive and clearly  ..... provide basic healthcare to all citizens." ",
                                "key_points": [
                                    "Positive economics",
                                    " cause-and-effect relationships",
                                    "authoritarian)"
                                ]
                            }
                        ]
                    }
                    """
                question_template = """
                    You are a skillful interviewer, and have vast knowledge on wide topics.You are generate question in json format. The question will have both multiple choice question and open ended questions.
                    Generate 3 MCQs and 3 open-ended questions with answers for the topic : {topic_content} similar to below format:
                    {question_examples}
                    Refer the following context to form better answer :
                   {question_content}
                """
                question_prompt = PromptTemplate(template = question_template, input_variables = ['topic_content', 'question_content'])
                question_chain = question_prompt | llm
                question_response = question_chain.invoke({'topic_content': topic_content, 'question_content': question_content,'question_examples': question_examples})
                #question_response_data = extract_questions_data(question_response)
                print("Questions Generated for ", topic_content)
                question_total_respose[topic_content] = question_response
            
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
