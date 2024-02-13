import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.output_parsers import StructuredOutputParser, ResponseSchema, PydanticOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field
from dotenv import load_dotenv,find_dotenv
import openai
from typing import List
import os
from serpapi import GoogleSearch

load_dotenv(find_dotenv())
openai.api_key = os.environ['OPENAI_API_KEY']


class GiftGuru(BaseModel):
    gift : List[str] = Field(description="The list of gifts to recommend based on the scenario.")
    reason: List[str] = Field(description = "Corresponding list of reasons to recommend the gift to the user. Should be atleast 20 words each")

def gift_recommendation(query):
    model = ChatOpenAI(model = 'gpt-3.5-turbo-1106', temperature =0.0)
    template = """You will be given a scenario from the user and you have to suggest \
    the best gifts as well as the reason for recommending the gift. There should be gifts which should be something that might be available on amazon \
    considering all the scenarios. The output should be a json with the following instructions:\n{format_instructions}

    Scenario: {scenario}"""
    pydantic_output_parser = PydanticOutputParser(pydantic_object = GiftGuru)
    prompt = ChatPromptTemplate.from_template(template)
    format_instructions = pydantic_output_parser.get_format_instructions()
    final_prompt = prompt.partial(format_instructions = format_instructions)
    chain = final_prompt | model | pydantic_output_parser
    response = chain.invoke({"scenario": query})

    return response


# Function to display the gift and reason as cards
def display_gift_cards(response):
    for i in range(len(response.gift)):
        st.write("## Gift:", response.gift[i])
        st.write("### Reason:", response.reason[i])
        
        if st.button(f"Get Items",key=response.gift[i]):
            get_amazon_links(response.gift[i])


def get_amazon_links(query):
    print("calling links in amazon")
    query = query +"-amazon"
    params = {
    "engine": "google",
    "q": query,
    "api_key": os.environ['GOOGLE_SERP_API_KEY']
    }
    search = GoogleSearch(params)
    results = search.get_dict()
    organic_results = results["organic_results"]
    print("google response",organic_results)
    for result in organic_results:
        st.image(result['thumbnail'])
        st.write(f" {result['title']}")
        st.markdown(f"[{result['link']}]({result['link']})")



def main():
    st.title("GiftGuru")
    st.write("Enter your prompt to get gift recommendations!")

    # Placeholder for user prompt
    user_prompt = st.text_input("Enter your prompt here:")
    res = gift_recommendation(user_prompt)
    if user_prompt:
        # Display recommended gifts and reasons
        display_gift_cards(res)

if __name__ == "__main__":
    main()
