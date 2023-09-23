# -*- coding: utf-8 -*-
"""
Created on Thu May  4 20:09:16 2023

@author: 21410
"""

from src import *
import json

with open("config.json", "r") as fread:
    config = json.load(fread)
bot = MerlinBot(config)
# bot._delete_index(config["PINECONE_INDEX_NAME"])
# bot.create_new_index(config["PINECONE_INDEX_NAME"])
# get list of files to upload
files = ["media/tcs-ar-22.pdf"]
# read pdfs into a list using pdf loader
pdfs = bot._load_pdfs(*files)
# split documents with token splitter
documents = bot._split_documents_by_tokentext(pdfs, chunk_size=500, chunk_overlap=100)
# send documents to pinecone index
bot._update_pinecone_index(documents)
# start conversation bot
bot._init_conversation_bot()
bot.perform_query("What is tcs")
bot.perform_query("consolidated statement for 2022")
bot.perform_query("who is the ceo")
