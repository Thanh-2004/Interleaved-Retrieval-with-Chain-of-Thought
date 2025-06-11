from typing import List, Dict
import json
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from utils import *
from llm import OpenRouterLLM

def load_corpus_to_knowledge_base(filepath: str) -> List[Dict]:
    knowledge_base = []
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()

    blocks = content.split("<endofpassage>")
    for idx, block in enumerate(blocks):
        if "Title:" in block and "Passage:" in block:
            try:
                title = block.split("Title:")[1].split("Passage:")[0].strip()
                passage = block.split("Passage:")[1].strip()
                knowledge_base.append({
                    "title": title,
                    "text": passage,
                    "embedding": None
                })
            except:
                continue
    return knowledge_base

class IRCoTSystem:
    def __init__(self, knowledge_base, prompt_template, my_llm, embedding_model="all-MiniLM-L6-v2"):
        self.knowledge_base = knowledge_base
        self.llm = my_llm
        self.embedding_model = SentenceTransformer(embedding_model)
        self.prompt_template = prompt_template

    def retrieve_initial(self, question, k=3):
        question_embedding = self.embedding_model.encode([question])[0]
        doc_embeddings = np.array([doc["embedding"] for doc in self.knowledge_base])
        sims = cosine_similarity([question_embedding], doc_embeddings)[0]
        top_ids = sims.argsort()[-k:][::-1]
        return [self.knowledge_base[i] for i in top_ids]

    def retrieve_with_cot(self, cot_sentence, k=1):
        cot_embedding = self.embedding_model.encode([cot_sentence])[0]
        doc_embeddings = np.array([doc["embedding"] for doc in self.knowledge_base])
        sims = cosine_similarity([cot_embedding], doc_embeddings)[0]
        top_ids = sims.argsort()[-k:][::-1]
        return [self.knowledge_base[i] for i in top_ids]

    def generate_next_cot(self, question, paragraphs, cot_so_far):
        formatted = ""
        for p in paragraphs:
            formatted += f"Title: {p['title']}\n{p['text']}\n\n"
        cot_text = "\n".join(cot_so_far)
        full_prompt = f"{formatted}Q: {question}\nA: {cot_text}\n"
        return self.llm.generate_response(self.prompt_template + full_prompt)

    def eval_evidence(self, output, target):
        f1 = cal_f1_score(output, target)
        acc = cal_accuracy_score(output, target)
        return f"F1: {f1:.2f}, Acc: {acc:.2f}"

    def answer_question(self, question_set, max_iterations=5):
        q = question_set["question"]
        gt = question_set["evidence"]
        paragraphs = self.retrieve_initial(q)
        cot_sentences = []

        for _ in range(max_iterations):
            next_cot = self.generate_next_cot(q, paragraphs, cot_sentences)
            cot_sentences.append(next_cot)
            if "answer is:" in next_cot.lower():
                break
            new_paras = self.retrieve_with_cot(next_cot)
            paragraphs.extend(new_paras)
            paragraphs = remove_duplicates(paragraphs)

        titles = extract_titles(paragraphs)
        eval_score = self.eval_evidence(titles, gt)
        return "\n".join(cot_sentences), eval_score


if __name__ == "__main__":
    corpus_path = "VDT2025_Multihop RAG dataset/multihoprag_corpus.txt"
    knowledge_base = load_corpus_to_knowledge_base(corpus_path)

    # Encode embeddings
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    for doc in knowledge_base:
        doc["embedding"] = embedding_model.encode(doc["text"])

    with open("VDT2025_Multihop RAG dataset/MultiHopRAG.json", "r", encoding="utf-8") as f:
        test_set = json.load(f)

    # Chuẩn hóa cho IRCoT
    questions = []
    for q in test_set:
        questions.append({
            "question": q["query"],
            "answer": q["answer"],
            "evidence": [e["title"] for e in q["evidence_list"]]
        })

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

    ircot = IRCoTSystem(knowledge_base=knowledge_base, my_llm=OpenRouterLLM("deepseek/deepseek-chat-v3-0324:free"), prompt_template=template)

    for i, qset in enumerate(questions[:3]):  # chạy 3 câu hỏi đầu tiên
        print(f"--- Question {i+1} ---")
        answer, score = ircot.answer_question(qset)
        print("Generated Reasoning:\n", answer)
        print("Evaluation:", score)
        print("\n")