from typing import List, Dict
import numpy as np
from sentence_transformers import SentenceTransformer
# from langchain.embeddings import HuggingFaceEmbeddings

from sklearn.metrics.pairwise import cosine_similarity
from utils import *

from llm import OpenRouterLLM
import json
import pandas as pd
from pymongo import MongoClient
import string




# embedding_model="sentence-transformers/all-mpnet-base-v2"

class IRCoTSystem: 
    def __init__(self, collection, prompt_template, my_llm, embedding_model="all-MiniLM-L6-v2"):
        self.collection = collection
        self.llm = my_llm 
        self.embedding_model = SentenceTransformer(embedding_model)
        
        self.eval = {
            'F1': [],
            "iterations": []
        }
        self.answer = []
        self.prompt_template = prompt_template
        
    # def retrieve_initial(self, question: str, k: int = 3) -> List[Dict]:
    #     # Initial retrieval using the question
    #     question_embedding = self.embedding_model.encode([question])[0]
    #     doc_embeddings = np.array([doc["embedding"] for doc in self.knowledge_base])
        
    #     similarities = cosine_similarity([question_embedding], doc_embeddings)[0]
    #     top_indices = similarities.argsort()[-k:][::-1]
        
    #     return [self.knowledge_base[i] for i in top_indices]

    def retrieve_initial(self, question:str, k: int = 5) -> List[Dict]:
        q_vec = self.embedding_model.encode([question])[0]
        q_norm = np.linalg.norm(q_vec)
        
        cursor = self.collection.find({}, {'title':1, 'chunk':1, 'embedding':1})
        
        best_per_title = {}  # title -> (score, chunk)
        for doc in cursor:
            title = doc['title']
            chunk = doc['chunk']
            emb = np.array(doc['embedding'], dtype=np.float32)
            score = np.dot(q_vec, emb) / (q_norm * np.linalg.norm(emb))
            
            if title not in best_per_title or score > best_per_title[title][2]:
                best_per_title[title] = (chunk, emb, score)
        
        results = [
            (title, chunk, emb, score)
            for title, (chunk, emb, score) in best_per_title.items()
        ]
        # results.sort(key=lambda x: x[0], reverse=True)

        # top_k = sorted(
        #     best_per_title.values(),
        #     key=lambda x: x[2],   # sort by score
        #     reverse=True
        # )[:k]

        results_sorted = sorted(results, key=lambda x: x[3], reverse=True)[:k]

        # 5. Build output list of dicts
        result_dict = [
            {
                "title": t,
                "text": txt,
                "embedding": emb,
                "score": score
            }
            for t, txt, emb, score in results_sorted
        ]

        return result_dict
        
    
    # def retrieve_with_cot(self, cot_sentence: str, k: int = 1) -> List[Dict]:
    #     # Retrieval using the latest CoT sentence
    #     cot_embedding = self.embedding_model.encode([cot_sentence])[0]
    #     doc_embeddings = np.array([doc["embedding"] for doc in self.knowledge_base])
        
    #     similarities = cosine_similarity([cot_embedding], doc_embeddings)[0]
    #     top_indices = similarities.argsort()[-k:][::-1]
        
    #     return [self.knowledge_base[i] for i in top_indices]

    def retrieve_with_cot(self, cot_sentence: str, k: int = 1) -> List[Dict]:
        q_vec = self.embedding_model.encode([cot_sentence])[0]
        q_norm = np.linalg.norm(q_vec)
        
        cursor = self.collection.find({}, {'title':1, 'chunk':1, 'embedding':1})
        
        best_per_title = {}  # title -> (score, chunk)
        for doc in cursor:
            title = doc['title']
            chunk = doc['chunk']
            emb = np.array(doc['embedding'], dtype=np.float32)
            score = np.dot(q_vec, emb) / (q_norm * np.linalg.norm(emb))
            
            if title not in best_per_title or score > best_per_title[title][2]:
                best_per_title[title] = (chunk, emb, score)
        
        results = [
            (title, chunk, emb, score)
            for title, (chunk, emb, score) in best_per_title.items()
        ]
        # results.sort(key=lambda x: x[0], reverse=True)

        # top_k = sorted(
        #     best_per_title.values(),
        #     key=lambda x: x[2],   # sort by score
        #     reverse=True
        # )[:k]

        results_sorted = sorted(results, key=lambda x: x[3], reverse=True)[:k]

        # 5. Build output list of dicts
        result_dict = [
            {
                "title": t,
                "text": txt,
                "embedding": emb,
                "score": score
            }
            for t, txt, emb, score in results_sorted
        ]

        return result_dict
    
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
    
    def eval_evidence(self, output, target, num_iter):
        # print("========================================")
        # print(f"Output: {output}")
        # print("========================================")
        # print(f"Target: {target}")
        # print("========================================")
        if len(target) == 0:
            precision = 1
            recall = 1
            f1_score = 1
            accuracy = 1
        else:
            precision, recall, f1_score = cal_f1_score(output, target)
            accuracy = cal_accuracy_score(output, target)
        # return f"F1 Score: {f1_score} \n Accuracy: {accuracy} \n Number of Iteration: {num_iter}"
        return (precision, recall, f1_score, accuracy, num_iter)

    def answer_question(self, question_set: dict, max_iterations: int = 10) -> str:
        # Initial retrieval
        question = question_set["question"]
        target = question_set["answer"]
        evidences = question_set["evidence"]


        paragraphs = self.retrieve_initial(question)
        cot_sentences = []
        num_iter = 0


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
        evidence_score = self.eval_evidence(titles, evidences, num_iter)

        answer = "\n".join(cot_sentences)

        return answer, evidence_score
        # return self.answer
        # return "\n".join(cot_sentences)


    def load_test_set(self, test_path):
        with open(test_path, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)

        test_set = []
        for item in raw_data:
            question = item['query']
            answer = item['answer']
            evidence_titles = [e['title'] for e in item.get('evidence_list', [])]
            
            test_set.append({
                "question": question,
                "answer": answer,
                "evidence": evidence_titles
            })

        # df = pd.DataFrame({
        #     "question": [d["question"] for d in test_set[:5]],
        #     "answer": [d["answer"] for d in test_set[:5]],
        #     "evidence": [", ".join(d["evidence"]) for d in test_set[:5]],
        # })

        # print(df.head(2))
        # tools.display_dataframe_to_user(name="Loaded Test Set Preview", dataframe=df)

        return test_set


if __name__ == "__main__":
    # Sample knowledge base (in a real system, this would be much larger)
    # model_name="llama-3.3-70b-versatile"
    # model_name="gpt-4o-mini"
    # model_name="gpt-3.5-turbo"
    model_name="deepseek/deepseek-r1-0528:free"
    my_llm = OpenRouterLLM(model_name)

    client = MongoClient('mongodb://localhost:27017')
    db = client['rag_db']
    collection = db['documents']

    
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")


    # Initialize the IRCoT system

    template = """
    Please generate your chain-of-thought reasoning step by step with provided evidences.  \n
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

    ircot = IRCoTSystem(collection=collection, 
                        my_llm=my_llm,
                        prompt_template=template
                        )

    # Answer a multi-step question
    # question = "How does IRCoT improve over traditional RAG approaches and what are its key components?"
    # question = "What is IRCoT in layman's term? Give me answer in simple language and in fewer words."

    def extract_answer_split(text: str) -> str:
        parts = text.split("Answer is:", 1)
        ans = parts[1] if len(parts) > 1 else text

        ans = ans.strip()

        m = re.match(r'^(yes|no)\b', ans, flags=re.IGNORECASE)
        if m:
            return m.group(1).lower()

        ans = ans.strip(string.punctuation + string.whitespace)
        return ans

    test_set = ircot.load_test_set(test_path='VDT2025_Multihop RAG dataset/MultiHopRAG.json')
    precision, recall, f1_score, accuracy, num_iter, match = 0, 0, 0, 0, 0, 0

    test_range = 20
    offset = 0
    for i, question_set in enumerate(test_set[offset : test_range]):
        print(f"Instance {i}: ")
        answer, evidence_score = ircot.answer_question(question_set)
        answer = extract_answer_split(answer)

        precision += evidence_score[0]
        recall += evidence_score[1]
        f1_score += evidence_score[2]
        accuracy += evidence_score[3]
        num_iter += evidence_score[4]

        if answer.lower() == question_set['answer'].lower():
            match += 1
        print(f"Question: {question_set["question"]}\n Answer: {answer}")
        print(f"Ground truth: {question_set['answer']}")

        print("Precision: ", precision/test_range)
        print("Recall: ", recall/test_range)
        print("F1 Score: ", f1_score/test_range)
        print("Accuracy: ", accuracy/test_range)
        print("Number of Iteration: ", num_iter/test_range)
        print("Match: ", match/test_range)
