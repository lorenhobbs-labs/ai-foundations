"""
Garmin R10 HelpBot
Study.com CS 311 Project #2

This program builds a small FAQ chatbot for the Garmin Approach R10 portable
launch monitor. The main point is not to build a huge commercial chatbot. The
point is to show a working NLP / LLM workflow using a product FAQ dataset,
LangChain, vector search, and an LLM response step.

The program is built to use OpenAI through LangChain when an OPENAI_API_KEY is
available. It also has a local fallback mode so the file can still run for
basic testing if no API key is set yet.
"""

# os is used to read environment variables such as OPENAI_API_KEY.
import os

# textwrap is used only to keep printed chatbot answers from running too wide.
import textwrap

# typing imports make the function inputs and outputs clearer.
from typing import List, Tuple

# pandas is used to load the FAQ dataset from the CSV file.
import pandas as pd

# dotenv loads variables from a .env file, which keeps the API key out of the code.
from dotenv import load_dotenv

# These are the LangChain document and embedding base classes.
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

# ChatPromptTemplate is used to build the prompt sent to the LLM.
from langchain_core.prompts import ChatPromptTemplate

# FAISS is the vector store used for similarity search over the FAQ dataset.
from langchain_community.vectorstores import FAISS

# ChatOpenAI and OpenAIEmbeddings are used when the user provides an OpenAI API key.
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# These scikit-learn tools are only used in the local fallback mode.
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize


class LocalTfidfEmbeddings(Embeddings):
    """
    A simple local embedding fallback.

    This is not meant to replace OpenAI embeddings. It is here so the program
    can still be demonstrated without an API key. For the assignment write-up,
    the main design is still LangChain + vector store + LLM API.
    """

    def __init__(self):
        # The vectorizer converts text into TF-IDF vectors.
        self.vectorizer = TfidfVectorizer(stop_words="english")

        # This flag tracks whether the vectorizer has already learned the FAQ vocabulary.
        self.is_fitted = False

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Convert FAQ documents into numeric vectors.
        """

        # Fit the vectorizer on the FAQ text the first time documents are embedded.
        matrix = self.vectorizer.fit_transform(texts)

        # Mark the vectorizer as fitted so queries use the same vocabulary later.
        self.is_fitted = True

        # Normalize the vectors so similarity search behaves more like cosine similarity.
        matrix = normalize(matrix)

        # FAISS expects normal Python lists, so convert the sparse matrix to dense lists.
        return matrix.toarray().tolist()

    def embed_query(self, text: str) -> List[float]:
        """
        Convert a user question into a numeric vector.
        """

        # If this happens, the setup step was missed. It should not happen in normal use.
        if not self.is_fitted:
            raise RuntimeError("The TF-IDF embedding model must be fitted before querying.")

        # Transform the user question using the same vocabulary learned from the FAQ data.
        matrix = self.vectorizer.transform([text])

        # Normalize the query vector for better similarity matching.
        matrix = normalize(matrix)

        # Return the single query vector as a list.
        return matrix.toarray()[0].tolist()


def load_faq_dataset(csv_file: str = "garmin_r10_faq_data.csv") -> pd.DataFrame:
    """
    Load and lightly clean the Garmin Approach R10 FAQ dataset.
    """

    # Read the CSV file into a pandas DataFrame.
    faq_data = pd.read_csv(csv_file)

    # These are the columns the rest of the program expects to find.
    required_columns = ["question", "answer", "category", "source"]

    # Make sure the CSV has the required structure. If not, stop with a clear message.
    missing_columns = [column for column in required_columns if column not in faq_data.columns]
    if missing_columns:
        raise ValueError(f"Missing required CSV columns: {missing_columns}")

    # Drop rows where the question or answer is blank. Those rows would not help retrieval.
    faq_data = faq_data.dropna(subset=["question", "answer"])

    # Fill optional metadata blanks with simple placeholder text.
    faq_data["category"] = faq_data["category"].fillna("General")
    faq_data["source"] = faq_data["source"].fillna("No source listed")

    # Remove leading and trailing spaces that sometimes get into CSV files.
    for column in required_columns:
        faq_data[column] = faq_data[column].astype(str).str.strip()

    # Reset the index after cleaning so row numbers are clean.
    faq_data = faq_data.reset_index(drop=True)

    return faq_data


def build_documents(faq_data: pd.DataFrame) -> List[Document]:
    """
    Convert each FAQ row into a LangChain Document.
    """

    documents = []

    # Loop through every FAQ row and make it searchable as one document.
    for row_number, row in faq_data.iterrows():
        # Put the question and answer together because both help similarity search.
        page_content = (
            f"Question: {row['question']}\n"
            f"Answer: {row['answer']}\n"
            f"Category: {row['category']}"
        )

        # Metadata keeps the source and category attached to the retrieved document.
        metadata = {
            "row": int(row_number),
            "category": row["category"],
            "source": row["source"],
            "question": row["question"],
        }

        documents.append(Document(page_content=page_content, metadata=metadata))

    return documents


def choose_embeddings() -> Tuple[Embeddings, str]:
    """
    Pick the embedding method.

    If an OpenAI API key is available, use OpenAI embeddings. If not, use the
    local TF-IDF fallback so the rest of the app can still be tested.
    """

    # Check whether the user has provided an OpenAI API key.
    openai_key = os.getenv("OPENAI_API_KEY")

    if openai_key:
        # This is the main assignment path: OpenAI embeddings through LangChain.
        return OpenAIEmbeddings(model="text-embedding-3-small"), "OpenAI embeddings"

    # This is the no-key demo path.
    return LocalTfidfEmbeddings(), "local TF-IDF fallback embeddings"


def build_vector_store(documents: List[Document]) -> Tuple[FAISS, str]:
    """
    Build the FAISS vector store from the FAQ documents.
    """

    # Pick either OpenAI embeddings or local fallback embeddings.
    embeddings, embedding_mode = choose_embeddings()

    # FAISS stores the embedded FAQ records and performs similarity search.
    vector_store = FAISS.from_documents(documents, embeddings)

    return vector_store, embedding_mode


def build_llm():
    """
    Build the LLM object when an OpenAI API key is available.
    """

    # If no API key is available, return None and let the app use fallback responses.
    if not os.getenv("OPENAI_API_KEY"):
        return None

    # The model can be changed from the .env file if needed.
    model_name = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    # Low temperature keeps the response more consistent and less creative.
    return ChatOpenAI(model=model_name, temperature=0.2)


def format_context(retrieved_documents: List[Document]) -> str:
    """
    Combine retrieved FAQ records into one context block for the LLM prompt.
    """

    context_blocks = []

    # Number the documents so the context is easier to read in debugging.
    for index, document in enumerate(retrieved_documents, start=1):
        block = (
            f"FAQ {index}\n"
            f"{document.page_content}\n"
            f"Source: {document.metadata.get('source', 'No source listed')}"
        )
        context_blocks.append(block)

    return "\n\n".join(context_blocks)


def answer_with_llm(question: str, context: str, llm) -> str:
    """
    Ask the LLM to answer using only the retrieved FAQ context.
    """

    # This prompt tells the model to stay grounded in the retrieved Garmin R10 FAQ content.
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are Garmin R10 HelpBot, a practical FAQ assistant for the Garmin "
                "Approach R10 portable golf launch monitor. Answer using only the "
                "provided context. If the context does not answer the question, say "
                "that the FAQ data does not clearly answer it and recommend checking "
                "Garmin Support or the owner manual. Keep the answer direct and useful.",
            ),
            (
                "human",
                "User question: {question}\n\nRetrieved FAQ context:\n{context}\n\nAnswer:",
            ),
        ]
    )

    # Build the final prompt with the user question and retrieved context.
    messages = prompt.format_messages(question=question, context=context)

    # Send the prompt to the LLM and return the text response.
    response = llm.invoke(messages)
    return response.content


def answer_without_api(question: str, retrieved_documents: List[Document]) -> str:
    """
    Local fallback response when no OpenAI API key is available.

    This does not pretend to be an LLM. It uses the top retrieved FAQ answer so
    the user can still test the dataset, vector store, and retrieval workflow.
    """

    # Use the best matching FAQ document as the basic answer.
    best_document = retrieved_documents[0]

    # Pull out the answer line from the stored document text.
    answer_line = ""
    for line in best_document.page_content.splitlines():
        if line.startswith("Answer: "):
            answer_line = line.replace("Answer: ", "", 1)
            break

    # Give a direct response and make it clear this is fallback mode.
    return (
        "I found the closest FAQ match. "
        f"{answer_line} "
        "Note: this run is using local fallback mode because no OpenAI API key was found."
    )


def answer_question(question: str, vector_store: FAISS, llm, top_k: int = 3) -> Tuple[str, List[Document]]:
    """
    Retrieve relevant FAQ entries and generate an answer.
    """

    # Search the vector store for the FAQ rows that are closest to the user question.
    retrieved_documents = vector_store.similarity_search(question, k=top_k)

    # Build the context block from the retrieved documents.
    context = format_context(retrieved_documents)

    # Use the LLM if an API key was available.
    if llm is not None:
        answer = answer_with_llm(question, context, llm)
    else:
        answer = answer_without_api(question, retrieved_documents)

    return answer, retrieved_documents


def print_sources(retrieved_documents: List[Document]) -> None:
    """
    Print the retrieved categories and sources so the answer is traceable.
    """

    print("\nRetrieved FAQ records:")

    # Show enough detail to prove retrieval is working without dumping the full dataset.
    for index, document in enumerate(retrieved_documents, start=1):
        print(f"  {index}. Category: {document.metadata.get('category')}")
        print(f"     Matched question: {document.metadata.get('question')}")
        print(f"     Source: {document.metadata.get('source')}")


def run_chatbot() -> None:
    """
    Main chatbot workflow.
    """

    # Load environment variables from .env if the user created that file.
    load_dotenv()

    print("Garmin R10 HelpBot")
    print("FAQ chatbot for the Garmin Approach R10 portable launch monitor")
    print()

    # Load and clean the FAQ dataset.
    faq_data = load_faq_dataset()

    # Convert FAQ rows into LangChain documents.
    documents = build_documents(faq_data)

    # Build the FAISS vector store.
    vector_store, embedding_mode = build_vector_store(documents)

    # Build the LLM if an OpenAI API key is available.
    llm = build_llm()

    # Report the setup so it is easy to screenshot for the assignment.
    print("Dataset and Model Setup")
    print(f"FAQ records loaded: {len(faq_data)}")
    print(f"Dataset features: {len(faq_data.columns)} columns")
    print(f"Embedding mode: {embedding_mode}")
    print(f"LLM mode: {'OpenAI API through LangChain' if llm is not None else 'local fallback mode'}")
    print()

    print("Ask a question about the Garmin Approach R10. Type 'quit' to exit.")
    print()

    # Keep asking questions until the user exits.
    while True:
        user_question = input("Question: ").strip()

        # Exit commands are handled here.
        if user_question.lower() in {"quit", "exit", "q"}:
            print("Goodbye.")
            break

        # Do not waste a retrieval call on a blank question.
        if not user_question:
            print("Please enter a question or type 'quit' to exit.")
            continue

        # Generate the answer and collect the retrieved records.
        answer, retrieved_documents = answer_question(user_question, vector_store, llm)

        # Print the chatbot response.
        print("\nAnswer:")
        print(textwrap.fill(answer, width=90))

        # Print traceability information for the report and screenshots.
        print_sources(retrieved_documents)
        print()


# This keeps the chatbot from running if the file is imported somewhere else.
if __name__ == "__main__":
    run_chatbot()
