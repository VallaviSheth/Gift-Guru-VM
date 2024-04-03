import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.output_parsers import StructuredOutputParser, ResponseSchema, PydanticOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field
from dotenv import load_dotenv, find_dotenv
import openai
from typing import List
import os
from serpapi import GoogleSearch
from PIL import Image
import io

load_dotenv(find_dotenv())
openai.api_key = os.environ['OPENAI_API_KEY']

class GiftGuru(BaseModel):
    gift: List[str] = Field(description="The list of gifts to recommend based on the scenario.")
    reason: List[str] = Field(description="Corresponding list of reasons to recommend the gift to the user. Should be at least 20 words each")

def gift_recommendation(query):
    model = ChatOpenAI(model='gpt-3.5-turbo-1106', temperature=0.0)
    template = """You will be given a scenario from the user and you have to suggest \
    the best gifts as well as the reason for recommending the gift. There should be gifts which should be something that might be available on amazon \
    considering all the scenarios. The output should be a json with the following instructions:\n{format_instructions}

    Scenario: {scenario}"""
    pydantic_output_parser = PydanticOutputParser(pydantic_object=GiftGuru)
    prompt = ChatPromptTemplate.from_template(template)
    format_instructions = pydantic_output_parser.get_format_instructions()
    final_prompt = prompt.partial(format_instructions=format_instructions)
    chain = final_prompt | model | pydantic_output_parser
    response = chain.invoke({"scenario": query})

    return response

def display_gift_cards(response):
    for i in range(len(response.gift)):
        st.write("## Gift:", response.gift[i])
        st.write("### Reason:", response.reason[i])

        if st.button(f"Get Items", key=response.gift[i]):
            get_amazon_links(response.gift[i])

def get_amazon_links(query):
    print("Fetching links from amazon")
    query = query + "-amazon"
    params = {
        "engine": "google",
        "q": query,
        "api_key": os.environ['GOOGLE_SERP_API_KEY']
    }
    search = GoogleSearch(params)
    results = search.get_dict()
    organic_results = results["organic_results"]
    for result in organic_results:
        st.image(result['thumbnail'])
        st.write(f"{result['title']}")
        st.markdown(f"[{result['link']}]({result['link']})")

def image_input(uploaded_file):
    image = Image.open(uploaded_file)
    params = {
        "engine": "google_reverse_image",
        "image_url": "https://cdn10.bigcommerce.com/s-f7f6vece63/products/4292/images/10077/61w7p56ZZwL._AC_SL1000___31345.1595441060.1280.1280.jpg?c=2",
        "api_key": os.environ['GOOGLE_SERP_API_KEY'],
        "q": 'Amazon'
    }
    search = GoogleSearch(params)
    results = search.get_dict()
    if 'inline_images' in results:
        inline_images = results['inline_images']
        for image in inline_images:
            source_link = image['source']
            original_image = image['original']
            title = image['title']
            st.write(f"### {title}")
            st.image(original_image, use_column_width=True)
            st.markdown(f"GET THIS GIFT FROM HERE: [{source_link}]({source_link})")
    else:
        st.write('# NO RESULTS FOUND')
    st.image(image, caption='Your Uploaded Image.', use_column_width=True)

def main():
    st.title("GiftGuru")
    st.write("Enter your prompt to get gift recommendations!")

    # Placeholder for user prompt
    user_prompt = st.text_input("Enter your prompt here:")
    print('USER PROMPT:', user_prompt)
    if len(user_prompt)>0:
        res = gift_recommendation(user_prompt)
        # Display recommended gifts and reasons
        display_gift_cards(res)

    # Image uploader
    uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])
    if uploaded_file is not None:
        image_input(uploaded_file)
        

if __name__ == "__main__":
    main()
