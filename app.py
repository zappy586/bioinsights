from flask import Flask, request, Response
from flask_cors import CORS
import time
from PIL import Image
import pytesseract
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
embeddings = HuggingFaceEmbeddings()
db3 = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
from werkzeug.utils import secure_filename
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.llms import LlamaCpp

callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
n_gpu_layers = 1  # Metal set to 1 is enough.
n_batch = 512  # Should be between 1 and n_ctx, consider the amount of RAM of your Apple Silicon Chip.
# Make sure the model path is correct for your system!
llm = LlamaCpp(
    model_path="openhermes-2.5-mistral-7b-16k.Q4_K_M.gguf",
    n_gpu_layers=n_gpu_layers,
    n_batch=n_batch,
    f16_kv=True,# MUST set to True, otherwise you will run into problem after a couple of calls
    temperature=0.70,
    max_tokens=1000,
    n_ctx=3000,
    # streaming=True,
    # callback_manager=callback_manager,
    # verbose=True,  # Verbose is required to pass to the callback manager
)

# llm = VLLM(
#     model = "openhermes-2.5-mistral-7b-16k.Q4_K_M.gguf",
#     max_new_tokens=512,
#     top_k=10,
#     top_p=0.95,
#     temperature=0.7
# )

# Initialize Flask app
app = Flask(__name__)
CORS(app)
@app.route("/", methods=['POST'])
def receive_data():
    print(request.form)
    data = request.form.get('name')
    docs = db3.similarity_search(data, k=8)
    query_res = docs[1::2]

    prompt = f"""|<system>|:You are Dr.insights, a friendly AI medical advisor and health assistant that answers the user's health related queries
in a direct and straightforward way using the DB_examples(Dont talk about these examples in the answers though) that has some previous example 
question and answer pairs to help you respond to the user. Sometimes you may get 'Hi' or such general query from the user in which case, you will ignore
the db examples and just greet them normally. Only use the db if the user_query is relevant otherwise ignore it.
Give a very long, point-wise and detailed answers to the user's query.|</system>|
|<user>|:{data}
|<examples>|:
(WARNING: DO NOT USE THESE DB EXAMPLES IF THE USER_QUERY IS NOT RELEVANT TO THEM)
{query_res}
|<assistant>|:

    """
    print(request.form.get("memory"))
    # return prompt
    def generate():
        for chunk in llm.stream(prompt):
            yield chunk.encode('utf-8')
    print(data)
    return Response(generate(), content_type='text/event-stream; charset=utf-8')

@app.route("/img", methods=['POST'])
def img_data():
    data = request.files.get('fileInput')
    image_data = pytesseract.image_to_string(Image.open(data))
    print(image_data)
    docs = db3.similarity_search(image_data, k=4)
    query_res = docs[1::2]

    prompt = f"""Context:You are Bioinsight, a friendly AI medical lab report analyzer that gives a detailed analysis of the user's health related reports like
blood reports, urine reports, etc in a direct and straightforward way using the DB_examples(Dont talk/reference about these examples in the answers though) that has some previous example
question and answer pairs to help you respond to the user. Only answer the query if it is relevant to medicine/healthcare. Otherwise just say "I am not
designed for this. Please upload a relevant lab report image"
Give the insights of the report in a long and detailed point-wise way.
Lab_report_data:
{image_data}
DB_examples:
(WARNING: DO NOT USE THESE DB EXAMPLES IF THE LAB_REPORT_DATA IS NOT RELEVANT TO THEM)
{query_res}
Assistant:
"""
    # return prompt
    def generate():
        for chunk in llm.stream(prompt):
            yield chunk.encode('utf-8')
    print(data)
    return Response(generate(), content_type='text/event-stream; charset=utf-8')

if __name__ == '__main__':
    app.run(debug=True, port=5000)
