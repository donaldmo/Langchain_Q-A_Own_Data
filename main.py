from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import WebBaseLoader

from langchain.document_loaders.generic import GenericLoader
from langchain.document_loaders.parsers import OpenAIWhisperParser
from langchain.document_loaders.blob_loaders.youtube_audio import YoutubeAudioLoader

from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter

import os

os.environ["OPENAI_API_KEY"] = "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"


def py_pdf_dataloader():
    loader = PyPDFLoader("docs/cs229_lectures/MachineLearning-Lecture01.pdf")
    pages = loader.load()

    print(len(pages))
    print(pages[0].page_content[0:500])
    print(pages[0].metadata)


def youtube_dataloader():
    url = "https://www.youtube.com/watch?v=jGwO_UgTS7I"
    save_dir = "docs/youtube/"

    loader = GenericLoader(
        YoutubeAudioLoader([url], save_dir),
        OpenAIWhisperParser()
    )

    docs = loader.load()

    print(docs[0].page_content[0:500])


def webbase_loader():
    # Use a markdown file from github page
    loader = WebBaseLoader(
        "https://github.com/donaldmo/flexbox/blob/main/README.md")

    docs = loader.load()
    print(docs[0])


if __name__ == "__main__":
    loader = PyPDFLoader("docs/cs229_lectures/MachineLearning-Lecture01.pdf")
    pages = loader.load()

    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=150,
        length_function=len
    )

    docs = text_splitter.split_documents(pages)

    print(len(docs))
    print(len(pages))
