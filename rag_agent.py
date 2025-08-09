"""RAG Agent module for Agentic RAG System"""
import os
import time
import re
import logging
from typing import TypedDict, Annotated, Sequence, List, Dict, Any
from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.schema import Document
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_core.prompts import PromptTemplate
from langgraph.graph import StateGraph, END

from config import Config

logger = logging.getLogger(__name__)


class ProductionAgentState(TypedDict):
    """Enhanced state definition for production RAG"""
    question: str
    generation: str
    documents: Sequence[str]
    document_scores: Sequence[float]
    retries: int
    queries: List[str]
    query_performance: Dict[str, int]
    retrieval_metadata: Dict[str, Any]
    processing_time: float
    confidence_level: str
    error_history: List[str]
    workflow_status: List[str]  # Add status tracking for UI display


class ProductionRAGAgent:
    """Production-ready RAG agent with OpenAI embeddings"""

    def __init__(self, config: Config, vectorstore_path: Path):
        self.config = config
        self.vectorstore_path = vectorstore_path

        # Initialize OpenAI models
        self.embeddings = OpenAIEmbeddings(
            openai_api_key=config.OPENAI_API_KEY,
            model=config.EMBEDDING_MODEL,
            dimensions=config.EMBEDDING_DIMENSIONS
        )

        self.llm = ChatOpenAI(
            openai_api_key=config.OPENAI_API_KEY,
            model=config.LLM_MODEL,
            temperature=config.LLM_TEMPERATURE
        )

        self.db = None
        self._load_vectorstore()

    def _load_vectorstore(self):
        """Load FAISS vectorstore with error handling"""
        try:
            index_path = self.vectorstore_path / "openai_embeddings_index"
            if index_path.exists():
                self.db = FAISS.load_local(
                    str(index_path),
                    self.embeddings,
                    allow_dangerous_deserialization=True
                )
                logger.info("✅ Vectorstore loaded successfully")
            else:
                logger.warning("⚠️ Vectorstore not found. Please create it first.")
        except Exception as e:
            logger.error(f"❌ Failed to load vectorstore: {e}")
            raise

    def adaptive_query_generation(self, state: ProductionAgentState) -> ProductionAgentState:
        """Generate adaptive queries based on retry count"""
        node_info = "🔄 NODE: Adaptive Query Generation"
        logger.info(node_info)
        
        # Add to workflow status for UI
        if 'workflow_status' not in state:
            state['workflow_status'] = []
        state['workflow_status'].append(node_info)

        question = state["question"]
        retries = state.get("retries", 0)

        strategies = {
            0: self._diverse_query_strategy,
            1: self._technical_query_strategy,
            2: self._broad_query_strategy
        }

        strategy = strategies.get(retries, strategies[2])

        try:
            queries = strategy(question)
            logger.info(f"✅ Generated {len(queries)} queries (strategy: {retries})")

            return {
                **state,
                "queries": queries,
                "retries": retries
            }

        except Exception as e:
            logger.error(f"❌ Query generation failed: {e}")
            return {
                **state,
                "queries": [question],
                "error_history": state.get("error_history", []) + [f"Query generation: {str(e)}"]
            }

    def _diverse_query_strategy(self, question: str) -> List[str]:
        """Generate diverse queries for initial attempt"""
        prompt = PromptTemplate(
            template="""Generate 3 diverse search queries from this question for document retrieval.

Each query should approach the topic differently:
1. Direct keyword extraction
2. Conceptual/semantic rephrasing
3. Specific aspect focus

Question: {question}

Return as JSON: {{"queries": ["query1", "query2", "query3"]}}""",
            input_variables=["question"]
        )

        chain = prompt | self.llm | JsonOutputParser()
        result = chain.invoke({"question": question})
        return result.get("queries", [question])

    def _technical_query_strategy(self, question: str) -> List[str]:
        """Generate technical queries for retry attempts"""
        prompt = PromptTemplate(
            template="""Generate 3 technical and specific queries from this question.

Focus on:
1. Technical terminology and domain-specific terms
2. Specific metrics, categories, or data points
3. Alternative technical phrasings

Question: {question}

Return as JSON: {{"queries": ["query1", "query2", "query3"]}}""",
            input_variables=["question"]
        )

        chain = prompt | self.llm | JsonOutputParser()
        result = chain.invoke({"question": question})
        return result.get("queries", [question])

    def _broad_query_strategy(self, question: str) -> List[str]:
        """Generate broad queries as fallback"""
        # Extract key terms from question
        words = re.findall(r'\b\w+\b', question.lower())
        important_words = [w for w in words if len(w) > 3 and
                          w not in ['what', 'how', 'when', 'where', 'why', 'the', 'and']]

        return [
            ' '.join(important_words[:3]) if len(important_words) >= 3 else question,
            ' '.join(important_words[:2]) if len(important_words) >= 2 else question,
            important_words[0] if important_words else question
        ]

    def multi_strategy_retrieval(self, state: ProductionAgentState) -> ProductionAgentState:
        """Advanced multi-strategy retrieval"""
        logger.info("🔍 NODE: Multi-Strategy Retrieval")

        queries = state["queries"]
        start_time = time.time()

        retrieval_configs = [
            {"name": "similarity", "search_type": "similarity", "k": 6},
            {"name": "similarity_relaxed", "search_type": "similarity", "k": 8},
            {"name": "mmr", "search_type": "mmr", "k": 4, "fetch_k": 15, "lambda_mult": 0.3}
        ]

        all_documents = []
        query_performance = {}

        for i, query in enumerate(queries):
            config = retrieval_configs[i % len(retrieval_configs)]

            try:
                retriever = self._create_retriever_safe(config)
                docs = retriever.invoke(query)

                # If no docs found, try with lower standards
                if not docs and config["search_type"] == "similarity":
                    logger.info(f"   🔄 No docs found, trying relaxed search...")
                    relaxed_retriever = self.db.as_retriever(
                        search_type="similarity",
                        search_kwargs={"k": 10}
                    )
                    docs = relaxed_retriever.invoke(query)

                all_documents.extend(docs)
                query_performance[query] = len(docs)

                logger.info(f"   ✅ Query {i+1}: {len(docs)} docs with {config['name']}")

            except Exception as e:
                logger.error(f"   ❌ Query {i+1} failed: {e}")
                query_performance[query] = 0

        # If still no documents, try emergency broad search
        if not all_documents:
            logger.info("🚨 Emergency broad search...")
            try:
                emergency_retriever = self.db.as_retriever(
                    search_type="similarity",
                    search_kwargs={"k": 15}
                )
                simple_query = queries[0].split()[:3]
                emergency_docs = emergency_retriever.invoke(" ".join(simple_query))
                all_documents.extend(emergency_docs)
                logger.info(f"   🆘 Emergency search found {len(emergency_docs)} docs")
            except Exception as e:
                logger.error(f"   ❌ Emergency search failed: {e}")

        # Advanced deduplication
        unique_documents = self._advanced_deduplication(all_documents)
        doc_texts = [doc.page_content for doc in unique_documents]

        processing_time = time.time() - start_time

        logger.info(f"📊 Retrieved {len(unique_documents)} unique docs from {len(all_documents)} total")

        return {
            **state,
            "documents": doc_texts,
            "query_performance": query_performance,
            "processing_time": state.get("processing_time", 0) + processing_time
        }

    def _create_retriever_safe(self, config: Dict):
        """Safe retriever creation with error handling"""
        try:
            if config["search_type"] == "similarity":
                return self.db.as_retriever(
                    search_type="similarity",
                    search_kwargs={"k": config["k"]}
                )
            elif config["search_type"] == "mmr":
                return self.db.as_retriever(
                    search_type="mmr",
                    search_kwargs={
                        "k": config["k"],
                        "fetch_k": config["fetch_k"],
                        "lambda_mult": config["lambda_mult"]
                    }
                )
            else:
                return self.db.as_retriever(
                    search_type="similarity",
                    search_kwargs={"k": 5}
                )
        except Exception as e:
            logger.error(f"Retriever creation failed: {e}, using fallback")
            return self.db.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 5}
            )

    def _advanced_deduplication(self, documents: List[Document], similarity_threshold: float = 0.85) -> List[Document]:
        """Advanced document deduplication"""
        if not documents:
            return []

        unique_docs = []
        seen_hashes = set()

        for doc in documents:
            content_hash = hash(doc.page_content[:200])
            if content_hash in seen_hashes:
                continue

            is_duplicate = False
            doc_words = set(doc.page_content.split()[:50])

            for existing_doc in unique_docs:
                existing_words = set(existing_doc.page_content.split()[:50])
                overlap = len(doc_words & existing_words)
                total = len(doc_words | existing_words)

                if total > 0 and overlap / total > similarity_threshold:
                    is_duplicate = True
                    break

            if not is_duplicate:
                unique_docs.append(doc)
                seen_hashes.add(content_hash)

        return unique_docs

    def intelligent_document_grading(self, state: ProductionAgentState) -> ProductionAgentState:
        """Grade documents with individual processing"""
        logger.info("⚖️ NODE: Intelligent Document Grading")

        question = state["question"]
        documents = state["documents"]

        if not documents:
            return {**state, "documents": [], "document_scores": [], "confidence_level": "low"}

        filtered_documents = []
        document_scores = []

        grading_prompt = PromptTemplate(
            template="""Rate this document's relevance to the question (0-10 scale).

Evaluation criteria:
1. DIRECT RELEVANCE (40%): Does it directly address the question?
2. COMPLETENESS (30%): Does it provide comprehensive information?
3. CONTEXT (20%): Is the context appropriate?
4. QUALITY (10%): Is information accurate and well-presented?

Question: {question}
Document: {document}

Return JSON: {{"overall_score": [0-10], "include_document": [true/false], "reasoning": "explanation"}}""",
            input_variables=["question", "document"]
        )

        chain = grading_prompt | self.llm | JsonOutputParser()

        for i, document in enumerate(documents):
            try:
                doc_content = document[:2000] + "..." if len(document) > 2000 else document
                result = chain.invoke({"question": question, "document": doc_content})

                score = float(result.get("overall_score", 0.0))
                include = result.get("include_document", False) and score >= self.config.MIN_DOCUMENT_SCORE

                document_scores.append(score)

                if include:
                    filtered_documents.append(document)
                    logger.info(f"   ✅ Document {i+1}: INCLUDED (Score: {score:.1f})")
                else:
                    logger.info(f"   ❌ Document {i+1}: EXCLUDED (Score: {score:.1f})")

                if i < len(documents) - 1:
                    time.sleep(self.config.RATE_LIMIT_DELAY)

            except Exception as e:
                logger.error(f"   ❌ Grading failed for document {i+1}: {e}")
                document_scores.append(3.0)

        confidence_level = self._calculate_confidence(document_scores)

        return {
            **state,
            "documents": filtered_documents,
            "document_scores": document_scores,
            "confidence_level": confidence_level
        }

    def _calculate_confidence(self, scores: List[float]) -> str:
        """Calculate confidence level based on scores"""
        if not scores:
            return "low"

        avg_score = sum(scores) / len(scores)
        max_score = max(scores)
        passed_docs = len([s for s in scores if s >= self.config.MIN_DOCUMENT_SCORE])

        if (max_score >= self.config.HIGH_CONFIDENCE_THRESHOLD and
            avg_score >= 7.0 and passed_docs >= 2):
            return "high"
        elif (max_score >= self.config.MEDIUM_CONFIDENCE_THRESHOLD and
              avg_score >= 5.0 and passed_docs >= 1):
            return "medium"
        else:
            return "low"

    def comprehensive_answer_generation(self, state: ProductionAgentState) -> ProductionAgentState:
        """Generate comprehensive answer from documents only"""
        logger.info("✍️ NODE: Document-Based Answer Generation")

        question = state["question"]
        documents = state["documents"]
        confidence_level = state.get("confidence_level", "medium")

        if not documents:
            return self._handle_no_documents(state)

        context = self._prepare_context(documents[:4])

        generation_prompt = PromptTemplate(
            template="""You are a document analyst providing answers STRICTLY from provided sources.

CRITICAL RULES:
1. Use ONLY information from the provided sources
2. Do NOT use general knowledge or external information
3. If information isn't in sources, state "This information is not available in the provided sources"
4. Always cite source numbers for each claim
5. If sources conflict, mention both perspectives

SOURCES:
{context}

QUESTION: {question}

Provide a comprehensive answer using only the source material and citing sources:""",
            input_variables=["context", "question"]
        )

        try:
            chain = generation_prompt | self.llm | StrOutputParser()
            answer = chain.invoke({"context": context, "question": question})

            metadata_summary = f"""

---
**ANALYSIS SUMMARY**
- **Sources Analyzed**: {len(documents)} documents
- **Confidence Level**: {confidence_level.upper()}
- **Response Type**: Document-only (no external knowledge)
"""

            final_answer = answer + metadata_summary

            return {
                **state,
                "generation": final_answer,
                "confidence_level": confidence_level
            }

        except Exception as e:
            logger.error(f"❌ Answer generation failed: {e}")
            return self._handle_generation_error(state, str(e))

    def _prepare_context(self, documents: List[str]) -> str:
        """Prepare structured context from documents"""
        context_parts = []
        for i, doc in enumerate(documents, 1):
            doc_content = doc[:2000] + "..." if len(doc) > 2000 else doc
            context_parts.append(f"=== SOURCE {i} ===\n{doc_content}")
        return "\n\n".join(context_parts)

    def _handle_no_documents(self, state: ProductionAgentState) -> ProductionAgentState:
        """Handle case when no documents are found"""
        return {
            **state,
            "generation": "I couldn't find relevant information in the provided documents to answer your question.",
            "confidence_level": "low"
        }

    def _handle_generation_error(self, state: ProductionAgentState, error: str) -> ProductionAgentState:
        """Handle generation errors"""
        return {
            **state,
            "generation": f"I encountered an error while generating the response: {error}",
            "confidence_level": "error"
        }

    def enhanced_decision_engine(self, state: ProductionAgentState) -> str:
        """Enhanced decision logic to prevent infinite loops"""
        documents = state.get("documents", [])
        document_scores = state.get("document_scores", [])
        retries = state.get("retries", 0)
        confidence_level = state.get("confidence_level", "low")

        max_score = max(document_scores) if document_scores else 0
        
        logger.info(f"🧠 Decision: {len(documents)} docs, max score: {max_score:.1f}, retries: {retries}")

        # CRITICAL: Always fallback after max retries to prevent recursion
        if retries >= self.config.MAX_RETRIES:
            logger.info(f"🚫 DECISION: MAX RETRIES REACHED → FALLBACK")
            return "fallback"

        # If no documents found at all and we've tried once, go to fallback
        if len(documents) == 0 and retries >= 1:
            logger.info(f"🚫 DECISION: NO DOCUMENTS FOUND AFTER RETRY → FALLBACK")
            return "fallback"

        # Generate if we have ANY documents with reasonable scores
        if len(documents) > 0 and max_score >= 2.5:  # Lowered threshold
            logger.info(f"✅ DECISION: SUFFICIENT CONTENT → GENERATE")
            return "generate"

        # Generate even with low scores if we have documents and retried
        if len(documents) > 0 and retries >= 1:
            logger.info(f"✅ DECISION: GENERATE WITH LOW CONFIDENCE AFTER RETRY")
            return "generate"

        # Retry only if we have attempts remaining
        elif retries < self.config.MAX_RETRIES:
            logger.info(f"🔄 DECISION: RETRY (attempt {retries + 1})")
            return "retry"

        # Fallback (safety net)
        else:
            logger.info(f"🚫 DECISION: SAFETY FALLBACK")
            return "fallback"

    def enhanced_fallback(self, state: ProductionAgentState) -> ProductionAgentState:
        """Enhanced fallback with helpful suggestions"""
        question = state["question"]
        retries = state.get("retries", 0)
        query_performance = state.get("query_performance", {})

        fallback_response = f"""I couldn't find sufficiently relevant information after {retries + 1} search attempts.

**Search Analysis:**
- Original Question: {question}
- Documents Found: {sum(query_performance.values())}
- Search Attempts: {retries + 1}

**Suggestions:**
1. Try more specific terminology from your document domain
2. Break complex questions into smaller parts
3. Ask about broader topics first, then narrow down
4. Use keywords that would appear in formal documents

Would you like to rephrase your question?"""

        return {
            **state,
            "generation": fallback_response,
            "confidence_level": "low"
        }

    def increment_retry(self, state: ProductionAgentState) -> ProductionAgentState:
        """Increment retry counter for next iteration"""
        return {
            **state,
            "retries": state.get("retries", 0) + 1
        }

    def build_workflow(self) -> StateGraph:
        """Build the complete production workflow"""
        logger.info("🏗️ Building production RAG workflow...")

        workflow = StateGraph(ProductionAgentState)

        # Add nodes
        workflow.add_node("query_generation", self.adaptive_query_generation)
        workflow.add_node("retrieval", self.multi_strategy_retrieval)
        workflow.add_node("grading", self.intelligent_document_grading)
        workflow.add_node("generation", self.comprehensive_answer_generation)
        workflow.add_node("fallback", self.enhanced_fallback)
        workflow.add_node("increment_retry", self.increment_retry)

        # Set entry point
        workflow.set_entry_point("query_generation")

        # Add edges
        workflow.add_edge("query_generation", "retrieval")
        workflow.add_edge("retrieval", "grading")
        workflow.add_edge("generation", END)
        workflow.add_edge("fallback", END)
        workflow.add_edge("increment_retry", "query_generation")

        # Add conditional edge
        workflow.add_conditional_edges(
            "grading",
            self.enhanced_decision_engine,
            {
                "generate": "generation",
                "retry": "increment_retry",
                "fallback": "fallback"
            }
        )

        return workflow.compile()


def create_production_rag_system(vectorstore_path: Path, config: Config):
    """Create production RAG system"""
    logger.info("🚀 Creating production RAG system with OpenAI embeddings...")

    try:
        agent = ProductionRAGAgent(config, vectorstore_path)
        workflow = agent.build_workflow()

        logger.info("✅ Production RAG system created successfully!")
        return workflow, agent

    except Exception as e:
        logger.error(f"❌ Failed to create RAG system: {e}")
        raise