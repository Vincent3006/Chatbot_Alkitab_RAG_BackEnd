# # rag_pipeline.py
# import os
# from tqdm import tqdm
# from langchain_community.vectorstores import Chroma
# from langchain_community.embeddings import HuggingFaceEmbeddings
# from langchain_core.documents import Document
# from langchain_core.prompts import PromptTemplate
# from langchain_ollama import ChatOllama
# import config
# from utils import extract_text_with_metadata, semantic_chunking
# CONDENSE_QUESTION_PROMPT_TEMPLATE = """
# Mengingat riwayat percakapan dan pertanyaan tindak lanjut, ubah pertanyaan tindak lanjut tersebut menjadi pertanyaan yang berdiri sendiri.
# Riwayat Percakapan:
# {chat_history}
# Pertanyaan Tindak Lanjut:
# {question}
# Pertanyaan yang Berdiri Sendiri:"""
# class CustomRAGPipeline:
#     def __init__(self):
#         print("1. Menginisialisasi Model dan Database...")
#         self.embedding_model = self._initialize_embeddings()
#         # --- TAMBAHKAN KODE DEBUGGING INI ---
#         print("\n" + "="*50)
#         print("DEBUGGING INFO - SKRIP RAG (KODE 2)")
#         try:
#             test_embedding_rag = self.embedding_model.embed_query("Ini adalah teks tes.")
#             print(f"Model yang berhasil dimuat: {self.embedding_model.model_name}")
#             print(f"DIMENSI EMBEDDING SAAT QUERY: {len(test_embedding_rag)}")
#         except Exception as e:
#             print(f"GAGAL MEMBUAT EMBEDDING DENGAN MODEL '{config.MODEL_NAME}': {e}")
#         print("="*50 + "\n")
#         # --- AKHIR DARI KODE DEBUGGING ---
#         self.vectorstore = self._load_or_create_vectorstore()
#         self.llm = self._initialize_llm()
#         self.chat_history = []
#         self.main_prompt = PromptTemplate(
#             template=config.PROMPT_TEMPLATE, 
#             input_variables=["context", "question"]
#         )
#         self.condense_question_prompt = PromptTemplate.from_template(CONDENSE_QUESTION_PROMPT_TEMPLATE)
#     # def _initialize_embeddings(self):
#     #     return HuggingFaceEmbeddings(
#     #         model_name=config.MODEL_NAME,
#     #         model_kwargs=config.MODEL_KWARGS,
#     #         encode_kwargs=config.exc
#     #     )
#     def _initialize_embeddings(self):
#         return HuggingFaceEmbeddings(
#         model_name=config.MODEL_NAME,
#         model_kwargs=config.MODEL_KWARGS,
#         encode_kwargs=config.ENCODE_KWARGS 
#     )
#     def _initialize_llm(self):
#         return ChatOllama(
#             model="qwen2.5:3b",
#             temperature=0
#         )
#     def _load_or_create_vectorstore(self):
#         COLLECTION_NAME = "semantic_chunks"
#         if os.path.exists(config.DB_NAME):
#             print(f"2. Database '{config.DB_NAME}' ditemukan. Memuat...")
#             return Chroma(
#                 persist_directory=config.DB_NAME,
#                 embedding_function=self.embedding_model,
#                 collection_name=COLLECTION_NAME
#             )
#         else:
#             print(f"2. Database '{config.DB_NAME}' tidak ditemukan. Membuat baru...")
#             if not os.path.exists(config.KNOWLEDGE_BASE_DIR):
#                 raise RuntimeError(f"Direktori '{config.KNOWLEDGE_BASE_DIR}' tidak ditemukan.")
#             pdf_files = [f for f in os.listdir(config.KNOWLEDGE_BASE_DIR) if f.endswith('.pdf')]
#             if not pdf_files:
#                 raise RuntimeError(f"Tidak ada PDF di '{config.KNOWLEDGE_BASE_DIR}'.")
#             all_final_chunks = []
#             for pdf_name in tqdm(pdf_files, desc="Memproses PDF untuk database"):
#                 pdf_path = os.path.join(config.KNOWLEDGE_BASE_DIR, pdf_name)
#                 extracted_data = extract_text_with_metadata(pdf_path)
#                 if not extracted_data: continue
#                 final_chunks = semantic_chunking(extracted_data, self.embedding_model, config.SIMILARITY_THRESHOLD)
#                 if final_chunks:
#                     for chunk in final_chunks: 
#                         chunk['metadata']['source_pdf'] = pdf_name
#                     all_final_chunks.extend(final_chunks)
#             if not all_final_chunks:
#                 raise RuntimeError("Tidak ada chunk yang berhasil dibuat.")
#             langchain_documents = [Document(page_content=c['document'], metadata=c['metadata']) for c in all_final_chunks]
#             vectorstore = Chroma.from_documents(
#                 documents=langchain_documents,
#                 embedding=self.embedding_model,
#                 persist_directory=config.DB_NAME,
#                 collection_name=COLLECTION_NAME
#             )
#             print(f"Jumlah dokumen: {vectorstore._collection.count()}")
#             return vectorstore
#     def _format_chat_history(self):
#         """Mengubah riwayat chat menjadi string."""
#         return "\n".join([f"Manusia: {q}\nAsisten: {a}" for q, a in self.chat_history])
#     def invoke(self, question: str):
#         """Fungsi utama untuk memproses permintaan chat."""
#         standalone_question = question
#         if self.chat_history:
#             print("   > Mengondensasi pertanyaan dengan riwayat chat...")
#             condense_chain = self.condense_question_prompt | self.llm
#             standalone_question = condense_chain.invoke({
#                 "chat_history": self._format_chat_history(),
#                 "question": question
#             }).content
#         print(f"   > Pertanyaan yang akan dicari: '{standalone_question}'")
#         print("   > Melakukan retrieval dokumen...")
#         retrieved_docs_with_scores = self.vectorstore.similarity_search_with_score(standalone_question, k=10)
#         print("\n" + "="*50)
#         print("--- HASIL RETRIEVAL DARI DATABASE ---")
#         print(f"Ditemukan {len(retrieved_docs_with_scores)} dokumen yang relevan:")
#         for i, (doc, score) in enumerate(retrieved_docs_with_scores):
#             print(f"\n[ Dokumen {i+1} ]")
#             print(f"  > Skor Jarak (Distance Score): {score:.4f} (Semakin rendah, semakin mirip)")
#             print(f"  > Sumber: {doc.metadata.get('source_pdf', 'N/A')}")
#             print(f"  > Rentang: {doc.metadata.get('source_range', 'N/A')}")
#             print(f"  > Konten: {doc.page_content[:200]}...") 
#         print("="*50 + "\n")
#         retrieved_docs = [doc for doc, score in retrieved_docs_with_scores]
#         print("   > Memformat konteks untuk LLM...")
#         document_prompt = PromptTemplate.from_template(config.DOCUMENT_PROMPT_TEMPLATE)
#         context_parts = []
#         for doc in retrieved_docs:
#             doc_metadata = {
#                 "source_range": doc.metadata.get('source_range', 'N/A'),
#                 "page_content": doc.page_content 
#             }
#             formatted_doc = document_prompt.format(**doc_metadata)
#             context_parts.append(formatted_doc)
#         context = "\n".join(context_parts)
#         final_prompt_for_llm = self.main_prompt.format(context=context, question=question)
#         print("\n" + "="*50)
#         print("--- PROMPT FINAL YANG DIKIRIM KE LLM ---")
#         print(final_prompt_for_llm)
#         print("="*50 + "\n")
#         print("   > Menghasilkan jawaban akhir...")
#         rag_chain = self.main_prompt | self.llm
#         answer = rag_chain.invoke({
#             "context": context,
#             "question": question
#         }).content
#         # self.chat_history.append((question, answer))
#         return {
#             "answer": answer,
#             "source_documents": retrieved_docs 
#         }
#     def retrieve_documents(self, question: str, k: int = 4):
#         """
#         Fungsi yang HANYA melakukan retrieval dokumen dari vectorstore
#         tanpa memanggil LLM.
#         """
#         print(f"   > Melakukan retrieval untuk pertanyaan: '{question}'")
#         retrieved_docs_with_scores = self.vectorstore.similarity_search_with_score(question, k=k)
        
#         results = []
#         for doc, score in retrieved_docs_with_scores:
#             results.append({
#                 "content": doc.page_content,
#                 "metadata": doc.metadata,
#                 "score": score
#             })
            
#         print(f"   > Ditemukan {len(results)} dokumen relevan.")
#         return results
    
# def initialize_rag_pipeline():
#     """Fungsi pembungkus untuk membuat instance pipeline."""
#     return CustomRAGPipeline()

# rag_pipeline.py

import os
from tqdm import tqdm
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
# --- PERUBAHAN 1: Ganti import dari Ollama ke Google Generative AI ---
from langchain_google_genai import ChatGoogleGenerativeAI 
import config
from utils import extract_text_with_metadata, semantic_chunking

CONDENSE_QUESTION_PROMPT_TEMPLATE = """
Mengingat riwayat percakapan dan pertanyaan tindak lanjut, ubah pertanyaan tindak lanjut tersebut menjadi pertanyaan yang berdiri sendiri.
Riwayat Percakapan:
{chat_history}
Pertanyaan Tindak Lanjut:
{question}
Pertanyaan yang Berdiri Sendiri:"""

class CustomRAGPipeline:
    def __init__(self):
        print("1. Menginisialisasi Model dan Database...")
        self.embedding_model = self._initialize_embeddings()
        # --- Kode debugging Anda tetap di sini ---
        print("\n" + "="*50)
        print("DEBUGGING INFO - SKRIP RAG (KODE 2)")
        try:
            test_embedding_rag = self.embedding_model.embed_query("Ini adalah teks tes.")
            print(f"Model yang berhasil dimuat: {self.embedding_model.model_name}")
            print(f"DIMENSI EMBEDDING SAAT QUERY: {len(test_embedding_rag)}")
        except Exception as e:
            print(f"GAGAL MEMBUAT EMBEDDING DENGAN MODEL '{config.MODEL_NAME}': {e}")
        print("="*50 + "\n")
        # --- Akhir dari kode debugging ---
        self.vectorstore = self._load_or_create_vectorstore()
        self.llm = self._initialize_llm() # Fungsi ini akan diubah
        self.chat_history = []
        self.main_prompt = PromptTemplate(
            template=config.PROMPT_TEMPLATE, 
            input_variables=["context", "question"]
        )
        self.condense_question_prompt = PromptTemplate.from_template(CONDENSE_QUESTION_PROMPT_TEMPLATE)

    def _initialize_embeddings(self):
        return HuggingFaceEmbeddings(
            model_name=config.MODEL_NAME,
            model_kwargs=config.MODEL_KWARGS,
            encode_kwargs=config.ENCODE_KWARGS 
        )

    # --- PERUBAHAN 2: Ganti implementasi fungsi ini ---
    def _initialize_llm(self):
        """
        Menginisialisasi LLM menggunakan Google Gemini.
        Pastikan GOOGLE_API_KEY sudah diatur di environment variable Anda.
        """
        print("   > Menginisialisasi LLM dengan Google Gemini...")
        return ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            temperature=0,
            google_api_key="AIzaSyC9y6r5IrJIlFEcdM5MzLd83mMswVYRHk4",
            convert_system_message_to_human=True # Penting untuk kompatibilitas prompt
        )

    def _load_or_create_vectorstore(self):
        COLLECTION_NAME = "semantic_chunks"
        if os.path.exists(config.DB_NAME):
            print(f"2. Database '{config.DB_NAME}' ditemukan. Memuat...")
            return Chroma(
                persist_directory=config.DB_NAME,
                embedding_function=self.embedding_model,
                collection_name=COLLECTION_NAME
            )
        else:
            # (Sisa dari fungsi ini tidak perlu diubah, tetap sama)
            print(f"2. Database '{config.DB_NAME}' tidak ditemukan. Membuat baru...")
            if not os.path.exists(config.KNOWLEDGE_BASE_DIR):
                raise RuntimeError(f"Direktori '{config.KNOWLEDGE_BASE_DIR}' tidak ditemukan.")
            pdf_files = [f for f in os.listdir(config.KNOWLEDGE_BASE_DIR) if f.endswith('.pdf')]
            if not pdf_files:
                raise RuntimeError(f"Tidak ada PDF di '{config.KNOWLEDGE_BASE_DIR}'.")
            
            all_final_chunks = []
            for pdf_name in tqdm(pdf_files, desc="Memproses PDF untuk database"):
                pdf_path = os.path.join(config.KNOWLEDGE_BASE_DIR, pdf_name)
                extracted_data = extract_text_with_metadata(pdf_path)
                if not extracted_data: continue

                final_chunks = semantic_chunking(extracted_data, self.embedding_model, config.SIMILARITY_THRESHOLD)
                
                if final_chunks:
                    for chunk in final_chunks: 
                        chunk['metadata']['source_pdf'] = pdf_name
                    all_final_chunks.extend(final_chunks)

            if not all_final_chunks:
                raise RuntimeError("Tidak ada chunk yang berhasil dibuat.")
            
            langchain_documents = [Document(page_content=c['document'], metadata=c['metadata']) for c in all_final_chunks]
            
            vectorstore = Chroma.from_documents(
                documents=langchain_documents,
                embedding=self.embedding_model,
                persist_directory=config.DB_NAME,
                collection_name=COLLECTION_NAME
            )
            print(f"Jumlah dokumen: {vectorstore._collection.count()}")
            return vectorstore

    # Sisa dari file (invoke, retrieve_documents, dll.) tidak perlu diubah sama sekali.
    # Kode di bawah ini tetap sama persis.

    def _format_chat_history(self):
        """Mengubah riwayat chat menjadi string."""
        return "\n".join([f"Manusia: {q}\nAsisten: {a}" for q, a in self.chat_history])

    def invoke(self, question: str):
        """Fungsi utama untuk memproses permintaan chat."""
        standalone_question = question
        if self.chat_history:
            print("   > Mengondensasi pertanyaan dengan riwayat chat...")
            condense_chain = self.condense_question_prompt | self.llm
            standalone_question = condense_chain.invoke({
                "chat_history": self._format_chat_history(),
                "question": question
            }).content
        
        print(f"   > Pertanyaan yang akan dicari: '{standalone_question}'")
        print("   > Melakukan retrieval dokumen...")
        retrieved_docs_with_scores = self.vectorstore.similarity_search_with_score(standalone_question, k=10)

        print("\n" + "="*50)
        print("--- HASIL RETRIEVAL DARI DATABASE ---")
        print(f"Ditemukan {len(retrieved_docs_with_scores)} dokumen yang relevan:")
        for i, (doc, score) in enumerate(retrieved_docs_with_scores):
            print(f"\n[ Dokumen {i+1} ]")
            print(f"  > Skor Jarak (Distance Score): {score:.4f} (Semakin rendah, semakin mirip)")
            print(f"  > Sumber: {doc.metadata.get('source_pdf', 'N/A')}")
            print(f"  > Rentang: {doc.metadata.get('source_range', 'N/A')}")
            print(f"  > Konten: {doc.page_content[:200]}...") 
        print("="*50 + "\n")

        retrieved_docs = [doc for doc, score in retrieved_docs_with_scores]
        
        print("   > Memformat konteks untuk LLM...")
        document_prompt = PromptTemplate.from_template(config.DOCUMENT_PROMPT_TEMPLATE)
        context_parts = []
        for doc in retrieved_docs:
            doc_metadata = {
                "source_range": doc.metadata.get('source_range', 'N/A'),
                "page_content": doc.page_content 
            }
            formatted_doc = document_prompt.format(**doc_metadata)
            context_parts.append(formatted_doc)
        
        context = "\n".join(context_parts)
        
        final_prompt_for_llm = self.main_prompt.format(context=context, question=question)

        print("\n" + "="*50)
        print("--- PROMPT FINAL YANG DIKIRIM KE LLM ---")
        print(final_prompt_for_llm)
        print("="*50 + "\n")
        
        print("   > Menghasilkan jawaban akhir...")
        rag_chain = self.main_prompt | self.llm
        answer = rag_chain.invoke({
            "context": context,
            "question": question
        }).content

        return {
            "answer": answer,
            "source_documents": retrieved_docs 
        }

    def retrieve_documents(self, question: str, k: int = 4):
        """
        Fungsi yang HANYA melakukan retrieval dokumen dari vectorstore
        tanpa memanggil LLM.
        """
        print(f"   > Melakukan retrieval untuk pertanyaan: '{question}'")
        retrieved_docs_with_scores = self.vectorstore.similarity_search_with_score(question, k=k)
        
        results = []
        for doc, score in retrieved_docs_with_scores:
            results.append({
                "content": doc.page_content,
                "metadata": doc.metadata,
                "score": score
            })
            
        print(f"   > Ditemukan {len(results)} dokumen relevan.")
        return results
    
def initialize_rag_pipeline():
    """Fungsi pembungkus untuk membuat instance pipeline."""
    return CustomRAGPipeline()