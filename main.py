import textbase
from textbase.message import Message
from textbase import models
import os
from typing import List
import openai 
from dotenv import load_dotenv
import pandas as pd
import json
from termcolor import colored
import langchain
langchain.verbose = False
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS

system_prompt = f"""
    Who You are: You are an online shopping platform chatbot .i given a list of products if we have the product which the user has asked then suggest them that otherwise suggest something a friend would suggest.
    
    
    Rememeber these rules:
    1) if what user needs does not exist in our product list then do keyword recommendation separated by commas.
    2) If the product exists then show in the given format: Name, product price, product description, product image.
    2)  Suggest only those product from the data which are most relevant to the question asked by the user do not unnecessarily populate. 
    3)  If the products which you think user should buy doesn't exist in the given data then give the general but correct recommendations  
    4)  Answer exactly what has been asked. Don't make assumptions, use the data given, if the given data can't answer then say that you can't answer, but don't assume.
    5)  Your response should be clear and concise and to the point to what user has asked.
    Now Begin generating your response
    """
x=5
chat_history =[{"role": "system", "content": system_prompt}]
#loading the product file 
def load_product_data():
    with open('product.json', 'r') as file:
        product_data = json.load(file)
    return product_data

products = load_product_data()
#loading the env for secret keys
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

def tokencount(list):
  count = 0
  for i in list:
    if isinstance(i, dict) and "content" in i and i["content"] is not None:
        count += len(i["content"])

  return count / 4

def removeOldMessages(list):
  while tokencount(list) > 6000:
    list.pop(1)

  return list

@textbase.chatbot("talking-bot")

def on_message(message_history: List[Message], state: dict = None):
   
    
    if state is None or "counter" not in state:
        state = {"counter": 0}
    else:
        state["counter"] += 1

    
    # Extract the user's input from the message_history
    user_input = message_history[-1].content
    print(x)
    #print(chat_history)
    # Get a bot reply using the get_reply function
    #bot_reply = get_reply(user_input, message_history, state)
    bot_reply= getAnswer(user_input,products)
   # Display the products in the bot's reply
    chat_history.append(bot_reply)
    #chat_history = removeOldMessages(chat_history) 
    return bot_reply, state


def getAnswer(question, dataToFindAnswerFrom, extraPrompt = None):
    
    
    print("KBase created")
    docs = knowledge_base.similarity_search(question)
    print("search term")        
    max_doc = max(docs, key=lambda doc: len(doc.page_content))
    context = max_doc.page_content
    
    msg= f""" Existing Product list: {context}
        
             {question}
         """
    
    chat_history.append({"role": "user", "content":msg})
    #chat_history.append({"role": "system", "content":f"product list which may fullfill what user has asked:{context}"})
    if extraPrompt is not None:
        prompt = f"{prompt} {extraPrompt}" 
    
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-16k",
        messages=chat_history,
        max_tokens=1000,
        temperature=0.5
    )
    answer = response.choices[0]["message"]["content"].strip()
    
    
    
    
    return answer

def getChunks(text):
    max_length = 2000
    original_string = text
    temp_string = ""
    strings_list = []

    for character in original_string:
        if len(temp_string) < max_length:
            temp_string += character
        else:
            strings_list.append(temp_string)
            temp_string = ""

    if temp_string:
        strings_list.append(temp_string)
        
    return strings_list
embeddings = OpenAIEmbeddings()
    
knowledge_base = FAISS.from_texts(getChunks(f"{products}"), embedding=embeddings)







    