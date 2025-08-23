# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.2
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown] id="pqHG6FcjlU4q"
# # Final exam: Python for Data Science

# %% [markdown] id="2G-05mXikUm3"
# ## Task 1: LLM integration

# %% [markdown] id="m8SbCGmB0esv"
# ### 1.1 Single Text Translation

# %% id="Hd1YTpeBfvG1"
import json
import random
import time
import textwrap
import numpy as np
import requests
import bs4
from bs4 import BeautifulSoup
from time import sleep
from IPython.display import Markdown
import warnings

warnings.filterwarnings("ignore")

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import google.generativeai as genai

# %% id="Gimbu3MRncO3"
genai.configure(api_key="Your_API")

# %% id="Tjcem43qr6Fe"
generator_config = {
    "temperature": 0,
    "top_k": 24,
    "top_p": 0.8,
    "max_output_tokens": 8192,
}

model = genai.GenerativeModel(
    model_name="models/gemini-1.5-flash-8b-latest",
    generation_config=generator_config,
)


# %% id="uqwty9TtkQ-C"
# Function to create the translation prompt
def create_translation_prompt(sentence, target_language):
    return (
        "Here is how to handle translations:  \n"
        "Input: \"Hello, my name is John\" in English. Target language: Vietnamese.  \n"
        "Output: \"Xin chào, tôi tên là John\".  \n\n"
        "Input: \"chào tôi tên Trang\" in Vietnamese. Target language: Vietnamese.  \n"
        "Output: \"chào tôi tên Trang\".  \n\n"
        "Now translate the following sentence:  \n"
        f"Input: \"{sentence}\". Target language: {target_language}.  \n"
        "Output:"
    )

# Clean up model's response to remove unwanted words
def clean_response(response_text):
    return response_text.replace("Output: ", "").strip()

# Single Text Translation
def translate_text(sentence, target_language):
    prompt = create_translation_prompt(sentence, target_language=target_language)

    try:
        response = model.generate_content(prompt)
        return clean_response(response.text)
    except Exception as e:
        print(f"Error translating text: {sentence}. Error: {e}")
        return f"Error: {e}"

# Multiple Text Translation
def translate_sentences(sentences, target_language="Vietnamese"):
    translations = []

    for sentence in sentences:
        prompt = create_translation_prompt(sentence, target_language)

        try:
            response = model.generate_content(prompt)
            translations.append(clean_response(response.text))
        except Exception as e:
            print(f"Error translating sentence: {sentence}. Error: {e}")
            translations.append(f"Error: {e}")

    return translations



# %% colab={"base_uri": "https://localhost:8080/", "height": 208} id="BBIUqsnV1NRI" outputId="15d9c6c4-52b1-4597-e608-9068c2d8262c"
list_language = ["Enlish", "French", "Vietnamese", "Portuguese", "German", "Thai", "Russian"]
for i in range(len(list_language)):
    print(f"{i + 1}. {list_language[i]}")

target_language = input("Nhập vào ngôn ngữ cần dịch dịch từ ngôn ngữ trên:\n")

idx_target_language = int(target_language) - 1

target_lang = list_language[idx_target_language]

single_text = "Hello, my name is Robert, I live in a rural area and have never been to a place as beautiful as this, tôi đến từ Hà Nội."

translated_text = translate_text(single_text, target_lang)
print(f"\nCâu sau khi dịch: {translated_text}")

# %% [markdown] id="JAz_9pPOCC3U"
# ## 1.2 Multiple Text Translation

# %% colab={"base_uri": "https://localhost:8080/", "height": 382} id="wNkXfz_6ptqi" outputId="1321b067-96c3-4eff-9e1e-8367e1962625"
sentenses = [
    "Helo, my name is Dũng, you can call me Daniel.",
    "I'm studying in University of Science.",
    "My english not good, vì thế mà tôi nói tiếng việt sẽ dễ hơn.",
    "저는 현재 데이터 과학을 공부하고 있습니다."
]

for i in range(len(list_language)):
    print(f"{i + 1}. {list_language[i]}")

idx_target_language = int(input("Nhập vào ngôn ngữ cần dịch dịch từ ngôn ngữ trên:\n")) - 1
target_language = list_language[idx_target_language]

translated_sentences = translate_sentences(sentenses, target_language)

for original, translated in zip(sentenses, translated_sentences):
    print(f"\nOriginal: {original}\nTranslated: {translated}")


# %% [markdown] id="oJQE1JCVCI5b"
# ## Task 2

# %% [markdown] id="5k_BlE1qCN6L"
# ### 2.1 Data Access and Indexing

# %% [markdown] id="BgpLMRTMjvEx"
# Parsing data

# %% colab={"base_uri": "https://localhost:8080/"} id="OR-mUQZ3zZiw" outputId="80d6818d-b0ed-4a16-c2d2-1e9086979f02"
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
import time
import random

# Configure Chrome options for headless browsing
chrome_options = Options()
chrome_options.add_argument("--headless")  # Run Chrome in headless mode
chrome_options.add_argument("--no-sandbox")  # Bypass OS security model
chrome_options.add_argument("--disable-dev-shm-usage")  # Overcome limited resource issues
chrome_options.add_argument("--disable-gpu")  # Disable GPU hardware acceleration

# Specify the Chrome driver path
chrome_options.add_argument("webdriver.chrome.driver=/usr/lib/chromium-browser/chromedriver")
driver = webdriver.Chrome(options=chrome_options)

# Open the target webpage
url = "https://www.presight.io/privacy-policy.html"
driver.get(url)

# Wait for a random duration to mimic human behavior
time.sleep(random.randint(5, 10))

# Locate the target div element using CSS selector
css_selector = "div.css-fugq39"
div_element = driver.find_element(By.CSS_SELECTOR, css_selector)

# Extract the inner HTML content of the div
div_html = div_element.get_attribute("innerHTML")

driver.quit()


# %% colab={"base_uri": "https://localhost:8080/"} id="uljuL0Gnx97e" outputId="9338b32c-6380-4861-ae9c-1da87b612dbb"
soup = BeautifulSoup(div_html, "html.parser")
pretty_html = soup.prettify()
print(pretty_html)


# %% id="_jojTtJcx95S"
# Dictionary to store structured data
indexed_data = {}
current_heading = None

# Iterate through all relevant HTML elements
for element in soup.find_all(["h2", "p", "i", "ul"]):

    # Detect a new heading (h2) and initialize a new section
    if element.name == "h2":
        current_heading = element.get_text(strip=True)
        if current_heading not in indexed_data:
            indexed_data[current_heading] = {
                "content": [],  # Stores paragraph content and list items
                "subheaders": []  # Stores subheaders (italic text with details)
            }

    # If the element is a paragraph (p), add its text to the current section
    elif element.name == "p" and current_heading:
        indexed_data[current_heading]["content"].append(element.get_text(strip=True))

    # If the element is an unordered list (ul), extract all list items
    elif element.name == "ul" and current_heading:
        list_items = [li.get_text(strip=True) for li in element.find_all("li")]
        indexed_data[current_heading]["content"].extend(list_items)

    # If the element is an italic tag (i), treat it as a subheader
    elif element.name == "i" and current_heading:
        subheader_title = element.get_text(strip=True)
        subheader_content = element.find_next_sibling("p").get_text(strip=True) if element.find_next_sibling("p") else ""
        subheader_list = []

        # Check if there's a list (ul) following the subheader and extract items
        sibling_ul = element.find_next_sibling("ul")
        if sibling_ul:
            subheader_list = [li.get_text(strip=True) for li in sibling_ul.find_all("li")]

        # Append subheader details to the current heading's subheaders list
        indexed_data[current_heading]["subheaders"].append({
            "Title": subheader_title,
            "Content": subheader_content,
            "List": subheader_list
        })


# %% id="Hg8fsdeBgouB"
# Convert the dictionary into a structured list format
indexed_list = [
    {
        "heading": heading,
        "content": " ".join(data["content"]),  # Merge paragraph texts and list items
        "subheaders": data["subheaders"]
    }
    for heading, data in indexed_data.items()
]


# %%
indexed_list

# %% [markdown]
# ### Indexed_List[7] và indexed_List[8] chính là các mục con của Indexed_list[6] (Tụi em check tại trang web). Nên cần chuyển 2 phần tử này vào subheaders của Indexed_list[6]

# %% [markdown]
#  {'heading': 'Access to Personal Information',
#   'content': '',
#   'subheaders': []},
#  {'heading': 'Accessing Your Personal Information',
#   'content': 'You have the right to access all of your personal information that we hold. Through the application, you can correct, amend, or append your personal information by logging into the application and navigating to your settings and profile.',
#   'subheaders': []},
#  {'heading': 'Automated Edit Checks',
#   'content': 'Presight employs automated edit checks to ensure that data entry fields are completed properly when collecting personal information. These edit checks help maintain data integrity and accuracy. You are encouraged to provide complete and valid information to ensure the smooth processing of their personal data.',
#   'subheaders': []},

# %%
# Merge the 7th and 8th headings into the subheaders of the 6th heading
indexed_list[6]['subheaders'].append({
    "Title": indexed_list[7]['heading'],
    "Content": indexed_list[7]['content'],
    "List": []
})

indexed_list[6]['subheaders'].append({
    "Title": indexed_list[8]['heading'],
    "Content": indexed_list[8]['content'],
    "List": []
})

# Remove redundant entries (7th and 8th headings)
del indexed_list[7:9]

# Clear content of the 4th heading
indexed_list[3]['content'] = ""

# %% [markdown] id="E2IilM93jrf4"
# indexing data

# %% colab={"base_uri": "https://localhost:8080/"} id="_ElRYkcSx90I" outputId="866ec1f6-7443-481c-b59e-aa9492c0101c"
# Print index list after clean
for item in indexed_list:
    print(f"Object: {item['heading']}")
    print(f"Content: {item['content']}")
    if item['subheaders']:
        print("Subheaders:")
        for sub in item['subheaders']:
            print(f"  - Title: {sub['Title']}")
            print(f"    Content: {sub['Content']}")
            if sub['List']:
                print(f"    List: {', '.join(sub['List'])}")
    print()


# %% colab={"base_uri": "https://localhost:8080/"} id="grToHWQD1CKi" outputId="c26ee701-95ea-4098-b600-e3fd91efb186"
with open('indexed_list.json', 'w', encoding='utf-8') as json_file:
    json.dump(indexed_list, json_file, ensure_ascii=False, indent=4)

print("Data has been saved to indexed_list.json")


# %% [markdown] id="D3tNzmwekACt"
# Embeddings

# %% id="VwphvdRq1CIV"
# Initial model for embeddings
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# Embedding for each item in index
content_list = [item for item in indexed_list]
embeddings = embedding_model.encode(content_list)
embeddings = np.array(embeddings)


# %% id="H0jqqScX1CFV"
# Initialize the Gemini model
model = genai.GenerativeModel('gemini-1.5-pro')

def get_query_embedding(query, embedding_model):
    return embedding_model.encode([query])

def find_best_match(query, embeddings, content_list, embedding_model, top_k=5):
    query_embedding = get_query_embedding(query, embedding_model)
    similarities = cosine_similarity(query_embedding, embeddings).flatten()

     # Get indices of the top-k most relevant content, sorted in descending order of similarity
    top_k_indices = np.argsort(similarities)[-top_k:][::-1]

    return [(content_list[idx], similarities[idx]) for idx in top_k_indices]



# %% id="VgE4RKRX1B_d"
def generate_answer_gemini(top_matches, question):
    # Create top 5 context
    context = "\n\n".join([f"- {content} (Similarity: {score:.4f})" for content, score in top_matches])

    # Prompt
    prompt = f"""
    Act as a professional assistant at company Presight in answering the question provided.
    Your job is to provide a clear and concise answer based only on the information provided in the context.
    Do not add any details or information beyond what is provided in the context.

    Context:
    {context}

    Question:
    {question}

    Requirements:
    1. Answer the question as detailed as possible from the provided context, make sure to provide all the details.
    2. If the answer is not in the context provided, just say "Your question is not in the company's database, please ask another question."\
    without any further answer, don't give wrong answers.
    3. If providing data in JSON format, transform the response to a human-friendly response.

    Answer:
    """

    # Response
    response = model.generate_content(prompt)
    return response.text


# %% colab={"base_uri": "https://localhost:8080/", "height": 332} id="JZwCDwTIRNLU" outputId="a3e963d6-c182-438b-a27a-e86f5b01331e"
# Test query
user_question = "What is policy"

start = time.time()
top_matches = find_best_match(user_question, embeddings, content_list, embedding_model)
answer = generate_answer_gemini(top_matches, user_question)
end = time.time()

print("User Question:", user_question)
print("\nTop 5 Best Matches:")
for i, (content, score) in enumerate(top_matches, 1):
    print(f"Top {i} (Score: {score:.4f}): {content}\n")

print("Answer:", answer)
print(f"\nExecution Time: {end - start:.4f} seconds")


# %% [markdown] id="WQCjrN-0jcJi"
# ### 2.2 Chatbot Development

# %% id="U0GKrb5uRNH3"
# Build chatbot
def chat(embeddings, content_list, embedding_model):
    print("🤖 Chào bạn!👋 Hãy đặt câu hỏi, tôi sẽ trả lời bạn.\n Gõ 'exit' để thoát.\n")

    while True:
        user_question = input("👤 Bạn: ")
        if user_question.lower() == "exit":
            print("\nTạm biệt!👋")
            break

        start = time.time()
        top_matches = find_best_match(user_question, embeddings, content_list, embedding_model)
        answer = generate_answer_gemini(top_matches, user_question)
        end = time.time()

        wrapped_answer = textwrap.fill(answer, width=120)
        print("\n🤖 Chatbot:\n" + wrapped_answer + "\n")
        print(f"⏳ Thời gian trả lời: {end - start:.4f} giây\n")


# %% colab={"base_uri": "https://localhost:8080/", "height": 711} id="pK_2YPHTRNEz" outputId="c5d98900-f0dd-4317-9411-84f84c24f8da"
chat(embeddings, content_list, embedding_model)
