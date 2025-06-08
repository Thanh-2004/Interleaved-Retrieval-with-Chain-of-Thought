from typing import List, Dict
import numpy as np
from sentence_transformers import SentenceTransformer
# from langchain.embeddings import HuggingFaceEmbeddings

from sklearn.metrics.pairwise import cosine_similarity
from utils import *

from llm import OpenRouterLLM

# embedding_model="sentence-transformers/all-mpnet-base-v2"

class IRCoTSystem: 
    def __init__(self, knowledge_base, prompt_template, my_llm, embedding_model="all-MiniLM-L6-v2"):
        self.knowledge_base = knowledge_base
        self.llm = my_llm 
        self.embedding_model = SentenceTransformer(embedding_model)
        
        self.eval = {
            'F1': [],
            "iterations": []
        }
        self.answer = []
        self.prompt_template = prompt_template
        
    def retrieve_initial(self, question: str, k: int = 3) -> List[Dict]:
        # Initial retrieval using the question
        question_embedding = self.embedding_model.encode([question])[0]
        doc_embeddings = np.array([doc["embedding"] for doc in self.knowledge_base])
        
        similarities = cosine_similarity([question_embedding], doc_embeddings)[0]
        top_indices = similarities.argsort()[-k:][::-1]
        
        return [self.knowledge_base[i] for i in top_indices]
    
    def retrieve_with_cot(self, cot_sentence: str, k: int = 1) -> List[Dict]:
        # Retrieval using the latest CoT sentence
        cot_embedding = self.embedding_model.encode([cot_sentence])[0]
        doc_embeddings = np.array([doc["embedding"] for doc in self.knowledge_base])
        
        similarities = cosine_similarity([cot_embedding], doc_embeddings)[0]
        top_indices = similarities.argsort()[-k:][::-1]
        
        return [self.knowledge_base[i] for i in top_indices]
    
    def generate_next_cot(self, question: str, paragraphs: List[Dict], cot_so_far: List[str]) -> str:
        # Construct prompt for generating the next CoT sentence
        prompt = self._construct_cot_prompt(question, paragraphs, cot_so_far)
        
        response = self.llm.generate_response(
            prompt=self.prompt_template + prompt,
            # max_tokens=100,
            # temperature=0.7
            )

        return response
    
    def _construct_cot_prompt(self, question: str, paragraphs: List[Dict], cot_so_far: List[str]) -> str:
        # Format paragraphs for the prompt
        formatted_paragraphs = ""
        for p in paragraphs:
            formatted_paragraphs += f"Title: {p['title']}\n{p['text']}\n\n"
        
        # Format CoT sentences so far
        cot_text = "\n".join(cot_so_far)
        
        # Construct the full prompt
        prompt = f"{formatted_paragraphs}Q: {question}\nA: {cot_text}\n"

        with open("prompt.txt", "a", encoding="utf-8") as f:
            f.write(prompt + "\n \n")

        return prompt
    
    def eval_evidence(self, output, target):
        f1_score = cal_f1_score(output, target)
        accuracy = cal_accuracy_score(output, target)
        return f"F1 Score: {f1_score} \n Accuracy: {accuracy}"

    def answer_question(self, question_set: dict, max_iterations: int = 10) -> str:
        # Initial retrieval
        question = question_set["question"]
        target = question_set["answer"]
        evidences = question_set["evidence"]


        paragraphs = self.retrieve_initial(question)
        cot_sentences = []
        num_iter = 1


        # Interleave reasoning and retrieval
        for it in range(max_iterations):
            # Generate next CoT sentence
            num_iter +=1
            next_sentence = self.generate_next_cot(question, paragraphs, cot_sentences)
            cot_sentences.append(next_sentence)


            with open("cot.txt", "a", encoding="utf-8") as f:
                f.write(f"Iteration {it}" + "\n")
                for cot in cot_sentences:
                    f.write(cot + "\n \n")
            
            # Check if answer is found
            if "answer is:" in next_sentence.lower():

                break
                
            # Retrieve more paragraphs based on the latest CoT sentence
            new_paragraphs = self.retrieve_with_cot(next_sentence)
            paragraphs.extend(new_paragraphs)

            #Check for duplications
            paragraphs = remove_duplicates(paragraphs)
        titles = extract_titles(paragraphs)
        evidence_score = self.eval_evidence(titles, evidences)

        answer = "\n".join(cot_sentences)

        return answer, evidence_score
        # return self.answer
        # return "\n".join(cot_sentences)


if __name__ == "__main__":
    # Sample knowledge base (in a real system, this would be much larger)
    # model_name="llama-3.3-70b-versatile"
    # model_name="gpt-4o-mini"
    # model_name="gpt-3.5-turbo"
    model_name="deepseek/deepseek-chat-v3-0324:free"
    my_llm = OpenRouterLLM(model_name)
    
    # It can be your pdf, website, wikipedia, or anything else.
    knowledge_base = [
        {
            "title": "IRCoT Framework Overview",
            "text": "The IRCoT framework represents a groundbreaking fusion of interleaved retrieval and chain-of-thought methodologies. This innovative approach creates a dynamic feedback loop between information gathering and reasoning processes. At its core, IRCoT begins with an initial query extraction phase. The system analyzes the user's question to identify key concepts and potential search terms. This preliminary analysis shapes the first retrieval iteration.",
            "embedding": None  # Will be populated by the embedding model
        },
        {
            "title": "Key Components of IRCoT",
            "text": "Key components of the IRCoT process include query understanding and decomposition, initial document retrieval, iterative reasoning steps, dynamic query refinement, evidence collection and synthesis, and continuous evaluation and adjustment. The magic happens in the continuous interplay between retrieval and reasoning. Each retrieved piece of information influences the next thought step, while each reasoning step guides subsequent retrieval actions.",
            "embedding": None
        },
        {
            "title": "IRCoT Implementation Challenges",
            "text": "Common challenges in IRCoT implementation include technical hurdles such as query expansion complexity, resource intensive processing, latency management, result coherence maintenance, and system scalability. Building an effective IRCoT system requires attention to optimization strategies. Performance can be enhanced through caching frequently accessed information, smart query decomposition, and parallel processing capabilities.",
            "embedding": None
        },
        {
            "title": "Prompting Techniques for IRCoT",
            "text": "Effective prompting forms the backbone of successful IRCoT implementation. Chain-of-Thought prompting must be carefully structured to guide both the retrieval and reasoning processes effectively. Core prompting principles include clear step sequencing, logical progression, context maintenance, adaptive refinement, error recovery, and result validation. The 'Let's think step-by-step' approach proves particularly effective when combined with retrieval operations.",
            "embedding": None
        },
        {
            "title": "IRCoT Performance Benefits",
            "text": "Using IRCoT with GPT3 substantially improves retrieval (up to 21 points) as well as downstream QA (up to 15 points) on four datasets: HotpotQA, 2WikiMultihopQA, MuSiQue, and IIRC. Similar substantial gains are observed in out-of-distribution (OOD) settings as well as with much smaller models such as Flan-T5-large without additional training. IRCoT reduces model hallucination, resulting in factually more accurate CoT reasoning.",
            "embedding": None
        },
        {
            "title": "RAG Overview",
            "text": "Retrieval-augmented generation (RAG) is a powerful combination of traditional LLMs with information retrieval systems. By accessing and incorporating relevant information from external sources, RAG models can produce more accurate and contextually relevant responses. RAG architecture means that you can constrain generative AI to your enterprise content sourced from vectorized documents and images.",
            "embedding": None
        },
        {
            "title": "Building AI Knowledge Bases",
            "text": "To build an AI knowledge base, start by defining your goals and scope. Then gather and preprocess data from relevant sources, including existing documents, FAQs, customer interactions, and other information. Preprocess the data by cleaning, organizing, and structuring it for AI analysis. Ensure data quality and accuracy to improve the effectiveness of your AI model. Select the right structure for organizing your content and implement appropriate AI models.",
            "embedding": None
        },
        {
            "title": "AI Knowledge Base Components",
            "text": "Key components of an AI knowledge base include machine learning models, natural language processing, and a data repository. Machine learning models enable the AI to learn from data, identify patterns, and make predictions with minimal human intervention. NLP allows AI to understand and interpret human language, which is essential for analyzing and responding to user queries. A centralized storage system keeps all the relevant data.",
            "embedding": None
        },
        {
            "title": "Amazon Bedrock Knowledge Bases",
            "text": "Amazon Bedrock Knowledge Bases is a fully managed capability with in-built session context management and source attribution that helps implement the entire RAG workflow from ingestion to retrieval and prompt augmentation. It automatically fetches data from sources such as Amazon S3, Confluence, Salesforce, SharePoint, or Web Crawler. Once the content is ingested, it converts it into blocks of text, the text into embeddings, and stores the embeddings in your vector database.",
            "embedding": None
        },
        {
            "title": "IRCoT vs Traditional RAG",
            "text": "Unlike traditional one-step retrieve-and-read approaches, IRCoT operates through an iterative process that alternates between extending CoT (reasoning) and expanding retrieved information. The system uses the question, previously collected paragraphs, and previously generated CoT sentences to generate the next reasoning step, then uses the last CoT sentence as a query to retrieve additional relevant paragraphs.",
            "embedding": None
        }
    ]    # Pre-compute embeddings for the knowledge base

    test_set = [
        {
            "question": "How does dynamic query refinement in IRCoT improve retrieval relevance, and what challenge does resource-intensive processing present?",
            "answer": "Dynamic query refinement in IRCoT uses the continuous interplay between chain-of-thought reasoning and information retrieval to focus each subsequent search on the most relevant concepts, improving retrieval relevance. However, resource-intensive processing can introduce high latency and scalability constraints in a production system.",
            "evidence": [
            "Key Components of IRCoT",
            "IRCoT Implementation Challenges"
            ]
        },
        {
            "question": "What are the primary steps required to build an AI knowledge base, and which AWS service can automatically handle ingestion and source attribution?",
            "answer": "To build an AI knowledge base you must define your goals and scope, gather and preprocess data from relevant sources, ensure data quality and structure, and implement appropriate machine learning and NLP models. Amazon Bedrock Knowledge Bases can automatically handle content ingestion, session context management, and source attribution.",
            "evidence": [
            "Building AI Knowledge Bases",
            "Amazon Bedrock Knowledge Bases"
            ]
        },
        {
            "question": "On which datasets does IRCoT achieve significant QA improvements, and which underlying framework enables its dynamic reasoning–retrieval loop?",
            "answer": "IRCoT achieves up to 15-point QA improvements on HotpotQA, 2WikiMultihopQA, MuSiQue, and IIRC datasets by leveraging an interleaved retrieval and chain-of-thought framework that creates a dynamic feedback loop between reasoning and retrieval steps.",
            "evidence": [
            "IRCoT Performance Benefits",
            "IRCoT Framework Overview"
            ]
        }
    ]
    
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    for doc in knowledge_base:
        doc["embedding"] = embedding_model.encode(doc["text"])



    # Initialize the IRCoT system

    template = """
    Please generate your chain-of-thought reasoning step by step with provided evidences.  \n
    Your response should not 
    Once you are provided enough evidence, conclude with a sentence beginning with ‘Answer is:’ to state the final answer. \n

    
    For example:
    Q: Jeremy Theobald and Christopher Nolan share what profession?
    A: Jeremy Theobald is an actor and producer. Christopher Nolan is a director, producer, and screenwriter. Therefore, they
    both share the profession of being a producer. So the answer is: producer.
    Q: What film directed by Brian Patrick Butler was inspired by a film directed by F.W. Murnau?
    A: Brian Patrick Butler directed the film The Phantom Hour. The Phantom Hour was inspired by the films such as Nosferatu
    and The Cabinet of Dr. Caligari. Of these Nosferatu was directed by F.W. Murnau. So the answer is: The Phantom Hour.
    Q: How many episodes were in the South Korean television series in which Ryu Hye−young played Bo−ra?
    A: The South Korean television series in which Ryu Hye−young played Bo−ra is Reply 1988. The number of episodes Reply
    1988 has is 20. So the answer is: 20 \n \n

    Evidence: \n
    """

    ircot = IRCoTSystem(knowledge_base=knowledge_base, 
                        my_llm=my_llm,
                        prompt_template=template
                        )

    # Answer a multi-step question
    # question = "How does IRCoT improve over traditional RAG approaches and what are its key components?"
    # question = "What is IRCoT in layman's term? Give me answer in simple language and in fewer words."

    question_set = test_set[0]
    answer, evidence_score = ircot.answer_question(question_set)
    
    print(f"Question: {question_set["question"]}\n{answer}")
    print(evidence_score)
