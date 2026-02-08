import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from typing import Any, AsyncIterable, Dict, List, Optional, Union
from langchain_core.messages import AIMessage, HumanMessage
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, UnstructuredPowerPointLoader
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain_core.documents import Document
from config.llm import gemini
from agents.base import BaseAgent
import asyncio
import time
import json
from pathlib import Path
from datetime import datetime, timedelta
import hashlib
from utils.logger import log
from concurrent.futures import ThreadPoolExecutor, as_completed

from dotenv import load_dotenv
import os
from dotenv import find_dotenv
load_dotenv(find_dotenv(), override=True)

from config.security import security_config


class RAGAgent(BaseAgent):
    """Enhanced RAG Agent for content-based file search with Google Generative AI embeddings."""
    
    def __init__(self, max_workers: int = 4, cache_ttl_hours: int = 24, google_api_key: str = None):
        super().__init__(
            agent_name='RAGAgent',
            description='Search for files based on their content using RAG with Google embeddings',
            content_types=['text', 'text/plain']
        )
        
        self.model = gemini
        self.max_workers = max_workers
        self.cache_ttl = timedelta(hours=cache_ttl_hours)
        
        # Initialize Google Generative AI embeddings
        # Use provided key or fall back to security config
        api_key = google_api_key or security_config.google_api_key
        
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model="models/gemini-embedding-001",
            google_api_key=api_key
        )
        
        self.vector_store = None
        self.retriever = None
        self.index_built = False
        self.last_index_time = None
        self.data_dir = None
        
        # Performance settings
        self.max_file_size_mb = 50
        self.similarity_threshold = 20  # Minimum similarity percentage
        
        # Simple file hash cache (no SQLite complexity)
        self._file_hash_cache = {}
        
        # Store the embeddings model type for cache validation
        self.embeddings_model = "google-gemini-embedding-001"
        
    @property
    def name(self):
        return self.agent_name
        
    def _get_cache_dir(self) -> str:
        """Get the cache directory for storing the vector store with Google embeddings."""
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        # Use a different cache directory for Google embeddings to avoid conflicts
        cache_dir = os.path.join(base_dir, 'vector_store_google')
        os.makedirs(cache_dir, exist_ok=True)
        return cache_dir
    
    def _get_file_hash(self, file_path: str) -> str:
        """Generate a hash of file content with simple caching."""
        try:
            stat = os.stat(file_path)
            cache_key = f"{file_path}:{stat.st_mtime}:{stat.st_size}"
            
            if cache_key in self._file_hash_cache:
                return self._file_hash_cache[cache_key]
            
            hasher = hashlib.md5()
            with open(file_path, 'rb') as f:
                for chunk in iter(lambda: f.read(8192), b""):
                    hasher.update(chunk)
            
            file_hash = hasher.hexdigest()
            self._file_hash_cache[cache_key] = file_hash
            
            # Simple cache cleanup - keep only last 100 entries
            if len(self._file_hash_cache) > 100:
                old_keys = list(self._file_hash_cache.keys())[:-50]
                for old_key in old_keys:
                    del self._file_hash_cache[old_key]
            
            return file_hash
        except Exception as e:
            log(f"Error hashing file {file_path}: {e}", level='warning')
            return ""
    
    def _get_data_dir_hash(self, data_dir: str) -> str:
        """Generate a hash of the data directory contents for change detection."""
        hasher = hashlib.md5()
        for root, _, files in os.walk(data_dir):
            for file in files:
                if file.startswith('.'):
                    continue
                file_path = os.path.join(root, file)
                try:
                    file_hash = self._get_file_hash(file_path)
                    hasher.update(file_hash.encode())
                except Exception as e:
                    log(f"Không thể hash file {file_path}: {e}", level='warning')
        return hasher.hexdigest()
    
    async def _is_index_up_to_date(self, data_dir: str) -> bool:
        """Check if the existing index is up to date and uses the correct embeddings model."""
        cache_dir = self._get_cache_dir()
        
        # Check if Chroma collection exists
        chroma_files = ['chroma.sqlite3', 'chroma-collection.parquet']
        if not any(os.path.exists(os.path.join(cache_dir, f)) for f in chroma_files):
            return False
        
        # Check index metadata
        index_info_file = os.path.join(cache_dir, 'index_info.json')
        if os.path.exists(index_info_file):
            try:
                with open(index_info_file, 'r') as f:
                    index_info = json.load(f)
                
                # Check if the embeddings model matches
                if index_info.get('embeddings_model') != self.embeddings_model:
                    log("Embeddings model changed, rebuilding index...")
                    return False
                
                # Check TTL
                index_time = datetime.fromisoformat(index_info['created_at'])
                if datetime.now() - index_time > self.cache_ttl:
                    return False
                
                # Quick directory modification check
                data_dir_mtime = os.path.getmtime(data_dir)
                if data_dir_mtime > index_info.get('data_dir_mtime', 0):
                    return False
                
                return True
            except Exception as e:
                log(f"Error reading index info: {e}", level='warning')
        
        return False
    
    async def _load_document(self, file_path: str) -> Optional[Document]:
        """Load document from file path (keeping original working logic)."""
        try:
            if not os.path.exists(file_path):
                log(f"File not found: {file_path}", level='error')
                return None
                
            # Check file size
            file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
            if file_size_mb > self.max_file_size_mb:
                log(f"Skipping large file ({file_size_mb:.1f}MB): {file_path}", level='warning')
                return None
                
            file_ext = os.path.splitext(file_path)[1].lower()
            
            try:
                if file_ext == '.pdf':
                    loader = PyPDFLoader(file_path)
                    docs = loader.load()
                    if not docs:
                        log(f"No content extracted from PDF: {file_path}", level='warning')
                        return None
                    combined_text = "\n\n".join([doc.page_content for doc in docs if doc.page_content.strip()])
                    if not combined_text.strip():
                        return None
                    return Document(
                        page_content=combined_text,
                        metadata={"source": file_path}
                    )
                    
                elif file_ext == '.docx':
                    loader = Docx2txtLoader(file_path)
                    doc = loader.load()
                    if not doc or not doc[0].page_content.strip():
                        log(f"No content extracted from DOCX: {file_path}", level='warning')
                        return None
                    return doc[0]
                    
                elif file_ext == '.pptx':
                    loader = UnstructuredPowerPointLoader(file_path)
                    docs = loader.load()
                    if not docs:
                        log(f"No content extracted from PPTX: {file_path}", level='warning')
                        return None
                    combined_text = "\n\n".join([doc.page_content for doc in docs if doc.page_content.strip()])
                    if not combined_text.strip():
                        return None
                    return Document(
                        page_content=combined_text,
                        metadata={"source": file_path}
                    )
                    
                elif file_ext == '.txt':
                    # Enhanced TXT reading with encoding detection
                    encodings = ['utf-8', 'utf-16', 'latin-1', 'cp1252', 'ascii']
                    for encoding in encodings:
                        try:
                            with open(file_path, 'r', encoding=encoding) as f:
                                content = f.read()
                                if content.strip():
                                    return Document(
                                        page_content=content,
                                        metadata={"source": file_path}
                                    )
                        except (UnicodeDecodeError, UnicodeError):
                            continue
                    log(f"Could not read TXT file with any encoding: {file_path}", level='warning')
                    return None
                    
                else:
                    log(f"Unsupported file type: {file_ext}", level='warning')
                    return None
                    
            except Exception as e:
                log(f"Error processing {file_path}: {str(e)}", level='error')
                return None
                
        except Exception as e:
            log(f"Unexpected error with {file_path}: {str(e)}", level='error')
            return None
    
    async def _process_files_parallel(self, file_paths: List[str]) -> List[Document]:
        """Process files in parallel for better performance."""
        documents = []
        
        def process_file(file_path: str) -> Optional[Document]:
            """Process a single file synchronously."""
            return asyncio.run(self._load_document(file_path))
        
        # Process files in batches to avoid overwhelming the system
        batch_size = self.max_workers * 2
        
        for i in range(0, len(file_paths), batch_size):
            batch = file_paths[i:i + batch_size]
            
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                # Submit all files in the batch
                future_to_file = {executor.submit(process_file, fp): fp for fp in batch}
                
                # Collect results
                for future in as_completed(future_to_file):
                    file_path = future_to_file[future]
                    try:
                        doc = future.result()
                        if doc:
                            documents.append(doc)
                    except Exception as e:
                        log(f"Error in parallel processing of {file_path}: {e}", level='error')
        
        return documents
    
    async def build_index(self, directory_path: str, force_rebuild: bool = False) -> bool:
        """Build or load the vector index with Google embeddings."""
        try:
            if not os.path.isdir(directory_path):
                log(f"Directory not found: {directory_path}", level='error')
                return False
                
            self.data_dir = directory_path
            
            # Check if we can use existing index
            if not force_rebuild and await self._is_index_up_to_date(directory_path):
                log("Loading existing RAG index from cache (Google embeddings)...")
                try:
                    self.vector_store = Chroma(
                        persist_directory=self._get_cache_dir(),
                        embedding_function=self.embeddings
                    )
                    self.retriever = self.vector_store.as_retriever(search_kwargs={"k": 5})
                    self.index_built = True
                    self.last_index_time = datetime.now()
                    log("RAG index loaded successfully from cache")
                    return True
                except Exception as e:
                    log(f"Error loading index from cache, will rebuild: {str(e)}", level='warning')
            
            log(f"Building new RAG index with Google embeddings from {directory_path}...")
            
            # Collect files (keeping original working logic)
            file_paths = []
            file_extensions = [".pdf", ".docx", ".pptx", ".txt"]
            
            for root, _, files in os.walk(directory_path):
                for file in files:
                    file_lower = file.lower()
                    if any(file_lower.endswith(ext) for ext in file_extensions):
                        file_path = os.path.join(root, file)
                        if os.path.isfile(file_path):
                            file_paths.append(file_path)
            
            if not file_paths:
                log("No supported files found to index", level='error')
                return False
                
            log(f"Found {len(file_paths)} files to process")
            
            # Process files - use parallel processing if many files, otherwise sequential
            if len(file_paths) > 10 and self.max_workers > 1:
                log("Using parallel processing for better performance...")
                documents = await self._process_files_parallel(file_paths)
            else:
                log("Using sequential processing...")
                documents = []
                processed_files = 0
                error_files = 0
                
                for file_path in file_paths:
                    log(f"Processing: {file_path}")
                    try:
                        doc = await self._load_document(file_path)
                        if doc:
                            # Ensure consistent metadata
                            doc.metadata.update({
                                "source": file_path,
                                "filename": os.path.basename(file_path),
                                "file_type": os.path.splitext(file_path)[1].lower()
                            })
                            documents.append(doc)
                            processed_files += 1
                        else:
                            error_files += 1
                    except Exception as e:
                        log(f"Error processing {file_path}: {str(e)}", level='error')
                        error_files += 1
                        continue
                
                log(f"Successfully processed {processed_files} files, {error_files} files had errors")
            
            if not documents:
                log("No valid documents found to index", level='error')
                return False
                
            log(f"Total documents to index: {len(documents)}")
            
            # Split documents (keeping original working approach)
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                length_function=len,
                separators=["\n\n", "\n", " ", ""]
            )
            
            splits = text_splitter.split_documents(documents)
            log(f"Split into {len(splits)} chunks")
            
            if not splits:
                log("No text chunks generated from documents", level='error')
                return False
            
            # Create vector store with Google embeddings
            log("Creating embeddings with Google Generative AI...")
            self.vector_store = Chroma.from_documents(
                documents=splits,
                embedding=self.embeddings,
                persist_directory=self._get_cache_dir()
            )
            
            # Save index metadata with embeddings model info
            index_info = {
                'created_at': datetime.now().isoformat(),
                'data_dir': directory_path,
                'data_dir_mtime': os.path.getmtime(directory_path),
                'num_documents': len(documents),
                'num_chunks': len(splits),
                'embeddings_model': self.embeddings_model,  # Track embeddings model
                'embeddings_provider': 'google_generative_ai'
            }
            
            index_info_file = os.path.join(self._get_cache_dir(), 'index_info.json')
            with open(index_info_file, 'w') as f:
                json.dump(index_info, f, indent=2)
            
            # Set up retriever (enhanced version)
            try:
                compressor = LLMChainExtractor.from_llm(self.model)
                self.retriever = ContextualCompressionRetriever(
                    base_compressor=compressor,
                    base_retriever=self.vector_store.as_retriever(search_kwargs={"k": 5})
                )
            except Exception as e:
                log(f"Error creating compressed retriever, using basic: {e}", level='warning')
                self.retriever = self.vector_store.as_retriever(search_kwargs={"k": 5})
            
            self.index_built = True
            self.last_index_time = datetime.now()
            log(f"RAG index built successfully with Google embeddings - {len(splits)} chunks")
            return True
            
        except Exception as e:
            log(f"Critical error in build_index: {str(e)}", level='error')
            import traceback
            log(traceback.format_exc(), level='error')
            return False
    
    async def _check_index(self) -> bool:
        """Check if index needs to be rebuilt."""
        if not self.index_built or not self.vector_store:
            if self.data_dir:
                return await self.build_index(self.data_dir)
            return False
            
        # Check if index is too old
        if self.last_index_time and datetime.now() - self.last_index_time > self.cache_ttl:
            log("Index TTL expired, rebuilding...")
            if self.data_dir:
                return await self.build_index(self.data_dir)
            return False
            
        return True
    
    def _format_detailed_results(self, search_results: List[Dict], show_details: bool = True) -> str:
        """Format search results according to the RAG search prompt template."""
        if not search_results:
            return "Không tìm thấy tài liệu nào phù hợp với yêu cầu của bạn."
        
        if not show_details:
            # Simple format for basic response
            unique_files = [result["file_path"] for result in search_results]
            if len(unique_files) == 1:
                return f"Tôi đã tìm thấy file: {unique_files[0]}"
            else:
                content = "Tôi đã tìm thấy các file sau:\n"
                for i, file in enumerate(unique_files, 1):
                    content += f"{i}. {file}\n"
                return content.strip()
        
        # Detailed format with rich information
        content = []
        
        if len(search_results) == 1:
            content.append(f"Tôi đã tìm thấy file: {search_results[0]['file_path']}")
            content.append("")  # Empty line
        else:
            content.append("Tôi đã tìm thấy các file sau:")
            for i, result in enumerate(search_results, 1):
                content.append(f"{i}. {result['file_path']}")
            content.append("")  # Empty line
        
        content.append("KẾT QUẢ CHI TIẾT:")
        content.append("")
        
        for i, result in enumerate(search_results, 1):
            filename = result.get("filename", os.path.basename(result["file_path"]))
            similarity = result.get("similarity", 0)
            file_path = result["file_path"]
            
            # File header with emoji
            content.append(f"📂 {filename} (Độ phù hợp: XẤP XỈ {similarity}%)")
            content.append(f"📍 Đường dẫn: {file_path}")
            content.append("🔍 Nội dung liên quan:")
            
            # Main snippet
            best_snippet = result.get("best_snippet", result.get("snippet", ""))
            if best_snippet:
                # Split snippet into bullet points if it's long
                snippet_lines = best_snippet.split('\n')
                for line in snippet_lines[:2]:  # Max 2 lines from main snippet
                    if line.strip():
                        content.append(f"- {line.strip()}")
            
            # Additional snippets
            additional_snippets = result.get("additional_snippets", [])
            for snippet in additional_snippets[:2]:  # Max 2 additional snippets
                if snippet and snippet != best_snippet:
                    snippet_lines = snippet.split('\n')
                    for line in snippet_lines[:1]:  # 1 line per additional snippet
                        if line.strip():
                            content.append(f"- {line.strip()}")
            
            if i < len(search_results):  # Add spacing between results
                content.append("")
        
        return "\n".join(content)
    
    def _extract_best_snippet(self, content: str, query_words: List[str], exact_phrases: List[str], query_original_words: List[str], snippet_length: int = 500) -> str:
        """Extract the most relevant snippet from content with enhanced multi-word and case-sensitive matching."""
        if len(content) <= snippet_length:
            return content
        
        # Find the position with most query word matches
        content_lower = content.lower()
        best_pos = 0
        best_score = 0
        
        # Sliding window to find best position with enhanced scoring
        window_size = snippet_length
        for i in range(0, len(content) - window_size, 50):
            window = content_lower[i:i + window_size]
            window_original = content[i:i + window_size]
            
            # Score for individual words (lowercase)
            word_score = sum(1 for word in query_words if word in window)
            
            # Extra score for exact phrases (lowercase)
            phrase_score = sum(3 for phrase in exact_phrases if phrase.lower() in window)
            
            # Extra score for case-sensitive matches
            case_score = sum(2 for word in query_original_words if word in window_original)
            
            # Extra score for exact case-sensitive phrases
            exact_case_score = sum(5 for phrase in exact_phrases if phrase in window_original)
            
            # Combined score with weights
            score = word_score + phrase_score + case_score + exact_case_score
            
            if score > best_score:
                best_score = score
                best_pos = i
        
        # Extract snippet around best position
        start = max(0, best_pos)
        end = min(len(content), start + snippet_length)
        snippet = content[start:end]
        
        # Clean up snippet boundaries
        if start > 0:
            space_pos = snippet.find(' ')
            if space_pos > 0 and space_pos < 50:
                snippet = snippet[space_pos + 1:]
        
        if end < len(content):
            last_period = snippet.rfind('.')
            if last_period > len(snippet) - 100:
                snippet = snippet[:last_period + 1]
            else:
                snippet += "..."
        
        return snippet.strip()
    
    async def search_by_content(self, query: str, top_k: int = 10) -> List[Dict]:
        """Search for documents using enhanced multi-word and case-sensitive matching with Google embeddings."""
        if not await self._check_index():
            log("Index not built or failed to build", level='error')
            return [{"error": "Index not built or failed to build."}]
        
        try:
            # Increase the number of documents to retrieve for better coverage
            # This helps ensure we don't miss any relevant documents
            if self.retriever:
                if hasattr(self.retriever, 'search_kwargs'):
                    self.retriever.search_kwargs["k"] = 30  # Increased from 20 to 30
                elif hasattr(self.retriever, 'base_retriever') and hasattr(self.retriever.base_retriever, 'search_kwargs'):
                    self.retriever.base_retriever.search_kwargs["k"] = 30  # Increased from 20 to 30
            
            # Use the retriever to get relevant documents
            docs = self.retriever.get_relevant_documents(query)
            
            if not docs:
                log("No documents found in search", level='warning')
                return []
            
            # Process results with file-level aggregation (enhanced)
            file_results = {}  # file_path -> best result
            
            # Extract core keywords from query by removing common words
            common_words = ["tìm", "file", "có", "nội", "dung", "liên", "quan", "đến", "chứa", "sau", "đó", "phân", "loại"]
            
            # Enhanced query processing for better multi-word matching
            # Split query into individual words for matching
            query_words = [w.strip() for w in query.lower().split() if w.strip() and w.lower() not in common_words]
            query_original_words = [w.strip() for w in query.split() if w.strip() and w.lower() not in common_words]  # Keep original case
            
            # Create exact phrase matches for better precision
            exact_phrases = []
            
            # Add the complete phrase in lowercase
            if len(query_words) > 1:
                # Add core keywords as a phrase
                core_phrase = " ".join(query_words)
                exact_phrases.append(core_phrase)
                
                # Add original case phrase for case-sensitive matching
                core_phrase_original = " ".join(query_original_words)
                exact_phrases.append(core_phrase_original)
                
                # Add adjacent word pairs for partial matching
                for i in range(len(query_words) - 1):
                    exact_phrases.append(f"{query_words[i]} {query_words[i+1]}")
            
            log(f"Search query words: {query_words}, exact phrases: {exact_phrases}")
            
            for doc in docs:
                file_path = doc.metadata.get("source")
                if not file_path or not os.path.exists(file_path):
                    continue
                
                # Extract snippet using enhanced method with all query components
                content = doc.page_content
                snippet = self._extract_best_snippet(content, query_words, exact_phrases, query_original_words, 300)
                
                # Enhanced similarity calculation
                content_lower = content.lower()
                
                # Basic word matching - count occurrences of each word
                word_matches = sum(content_lower.count(word) for word in query_words)
                
                # Exact phrase matching with increased weights
                phrase_matches = 0
                for phrase in exact_phrases:
                    # Check lowercase version
                    phrase_lower = phrase.lower()
                    if phrase_lower in content_lower:
                        # Count occurrences and multiply by weight
                        phrase_matches += content_lower.count(phrase_lower) * 3
                    
                    # Check case-sensitive version (higher weight)
                    if phrase in content:
                        phrase_matches += content.count(phrase) * 5
                
                # Case-sensitive word matching (e.g., "KẾ HOẠCH" exactly as typed)
                case_sensitive_matches = sum(content.count(word) * 2 for word in query_original_words)
                
                # Combined score with weights
                total_matches = word_matches + phrase_matches + case_sensitive_matches
                
                # Calculate final similarity score (50-95% range)
                # Higher ceiling for better matches
                max_possible_matches = (len(query_words) * 2) + (len(exact_phrases) * 5) + (len(query_original_words) * 2)
                similarity_percent = min(95, max(50, int((total_matches / max(max_possible_matches, 1)) * 100)))
                
                # If file already exists, keep the one with better content match
                if file_path in file_results:
                    if total_matches > file_results[file_path].get("total_matches", 0):
                        file_results[file_path].update({
                            "similarity": similarity_percent,
                            "best_snippet": snippet,
                            "word_matches": word_matches,
                            "phrase_matches": phrase_matches,
                            "case_matches": case_sensitive_matches,
                            "total_matches": total_matches
                        })
                    # Add additional snippets
                    if snippet not in file_results[file_path].get("additional_snippets", []):
                        if "additional_snippets" not in file_results[file_path]:
                            file_results[file_path]["additional_snippets"] = []
                        if len(file_results[file_path]["additional_snippets"]) < 2:
                            file_results[file_path]["additional_snippets"].append(snippet)
                else:
                    # New file
                    file_results[file_path] = {
                        "file_path": file_path,
                        "filename": doc.metadata.get("filename", os.path.basename(file_path)),
                        "file_type": os.path.splitext(file_path)[1].lower(),
                        "best_snippet": snippet,
                        "similarity": similarity_percent,
                        "additional_snippets": [],
                        "word_matches": word_matches,
                        "phrase_matches": phrase_matches,
                        "case_matches": case_sensitive_matches,
                        "total_matches": total_matches,
                        "page": doc.metadata.get("page", 1) if "page" in doc.metadata else None
                    }
            
            # Convert to list and sort by total matches (better relevance)
            results = list(file_results.values())
            
            # Filter out results with very low match scores to ensure consistency
            # This helps prevent borderline matches from appearing in some queries but not others
            min_match_threshold = 2  # Minimum number of matches required
            results = [r for r in results if r.get("total_matches", 0) >= min_match_threshold]
            
            # Enhanced sorting logic: prioritize exact matches first, then total matches
            # This ensures files with exact phrase matches appear at the top
            results.sort(key=lambda x: (
                x.get("phrase_matches", 0) > 0,  # First priority: has phrase match
                x.get("case_matches", 0) > 0,    # Second priority: has case match
                x.get("total_matches", 0),       # Third priority: total match score
                x.get("filename", "")            # Fourth priority: filename (for stable sorting)
            ), reverse=True)
            
            # Include more results for multi-word queries to ensure comprehensive coverage
            if len(query_words) > 1:
                # Keep more results for multi-word queries
                results = results[:min(top_k * 2, len(results))]
            else:
                # Standard limit for single-word queries
                results = results[:top_k]
            
            log(f"Found {len(results)} relevant documents across multiple files")
            return results
            
        except Exception as e:
            log(f"Error in search_by_content: {str(e)}", level='error')
            import traceback
            log(traceback.format_exc(), level='error')
            return [{"error": str(e)}]
    
    async def invoke(self, query: str, session_id: str = None, show_detailed_results: bool = False) -> Dict[str, Any]:
        """Enhanced invoke with detailed formatting support."""
        try:
            if not self.index_built:
                if not self.data_dir:
                    return {
                        'response_type': 'error',
                        'content': 'Thư mục dữ liệu chưa được thiết lập. Vui lòng gọi build_index() trước.',
                        'is_task_complete': True,
                        'require_user_input': False
                    }
                
                success = await self.build_index(self.data_dir)
                if not success:
                    return {
                        'response_type': 'error',
                        'content': 'Không thể xây dựng chỉ mục tìm kiếm với Google embeddings.',
                        'is_task_complete': True,
                        'require_user_input': False
                    }
            
            # Normalize query by removing common words to improve consistency
            original_query = query
            
            # Log the original query for debugging
            log(f"RAGAgent processing query: '{original_query}'")
            
            # Search with higher limit to find more files (25 for multi-word queries)
            top_k = 25 if len(query.split()) > 1 else 15
            search_results = await self.search_by_content(original_query, top_k=top_k)
            
            # Log the number of results found
            log(f"RAGAgent found {len(search_results)} results for query: '{original_query}'")
            if search_results:
                log(f"Top result: {search_results[0].get('filename', 'unknown')}, score: {search_results[0].get('total_matches', 0)}")
            
            if not search_results:
                return {
                    'response_type': 'message',
                    'content': 'Không tìm thấy tài liệu nào phù hợp với yêu cầu của bạn.',
                    'is_task_complete': True,
                    'require_user_input': False
                }
            
            if len(search_results) == 1 and "error" in search_results[0]:
                return {
                    'response_type': 'error',
                    'content': f'Lỗi khi tìm kiếm: {search_results[0]["error"]}',
                    'is_task_complete': True,
                    'require_user_input': False
                }
            
            # Extract unique files from search results
            unique_files = []
            for result in search_results:
                file_path = result.get("file_path")
                if file_path and os.path.exists(file_path) and file_path not in unique_files:
                    unique_files.append(file_path)
            
            if not unique_files:
                return {
                    'response_type': 'message',
                    'content': 'Không tìm thấy tài liệu nào phù hợp với yêu cầu của bạn.',
                    'is_task_complete': True,
                    'require_user_input': False
                }
            
            # Format results based on requirement
            content = self._format_detailed_results(search_results, show_detailed_results)
            
            return {
                'response_type': 'message',
                'content': content,
                'file_paths': unique_files,
                'search_details': search_results,
                'is_task_complete': True,
                'require_user_input': False,
                'metadata': {
                    'query': query,
                    'num_results': len(search_results),
                    'num_files': len(unique_files),
                    'search_time': time.time(),
                    'detailed_format': show_detailed_results,
                    'embeddings_provider': 'google_generative_ai'
                }
            }
            
        except Exception as e:
            log(f"Lỗi khi tìm kiếm nội dung: {str(e)}", level='error')
            return {
                'response_type': 'error',
                'content': f'Có lỗi xảy ra khi tìm kiếm nội dung: {str(e)}',
                'is_task_complete': True,
                'require_user_input': False
            }
    
    async def ainvoke(self, input_data: Union[str, Dict], config: Optional[Dict] = None, **kwargs) -> Dict[str, Any]:
        """Enhanced ainvoke (keeping original working interface)."""
        try:
            # Extract query from input_data (keeping original working logic)
            if isinstance(input_data, dict):
                if 'input' in input_data:
                    query = input_data['input']
                elif 'query' in input_data:
                    query = input_data['query']
                elif 'messages' in input_data and input_data['messages']:
                    last_message = input_data['messages'][-1]
                    if hasattr(last_message, 'content'):
                        query = last_message.content
                    elif isinstance(last_message, dict) and 'content' in last_message:
                        query = last_message['content']
                    else:
                        query = str(last_message)
                else:
                    query = str(input_data)
            else:
                query = str(input_data)
            
            # Get session_id (keeping original working logic)
            session_id = None
            if config and 'configurable' in config and 'session_id' in config['configurable']:
                session_id = config['configurable']['session_id']
            elif config and 'configurable' in config and 'thread_id' in config['configurable']:
                session_id = config['configurable']['thread_id']
            elif 'session_id' in kwargs:
                session_id = kwargs['session_id']
            
            result = await self.invoke(query, session_id)
            
            # Format response for LangChain (keeping original working logic)
            if isinstance(input_data, dict) and 'messages' in input_data:
                if result.get('response_type') == 'error':
                    return {"messages": [AIMessage(content=result.get('content', 'Có lỗi xảy ra'))]}
                return {"messages": [AIMessage(content=result.get('content', ''))]}
            
            return result
            
        except Exception as e:
            log(f"Lỗi trong ainvoke: {str(e)}", level='error')
            return {
                'response_type': 'error',
                'content': f'Có lỗi xảy ra: {str(e)}',
                'is_task_complete': True,
                'require_user_input': False
            }
    
    # Additional utility methods
    def get_index_stats(self) -> Dict[str, Any]:
        """Get statistics about the current index."""
        if not self.index_built:
            return {'status': 'not_built'}
        
        try:
            index_info_file = os.path.join(self._get_cache_dir(), 'index_info.json')
            if os.path.exists(index_info_file):
                with open(index_info_file, 'r') as f:
                    index_info = json.load(f)
            else:
                index_info = {}
            
            return {
                'status': 'built',
                'index_built_at': self.last_index_time.isoformat() if self.last_index_time else None,
                'data_directory': self.data_dir,
                'num_documents': index_info.get('num_documents', 0),
                'num_chunks': index_info.get('num_chunks', 0),
                'cache_ttl_hours': self.cache_ttl.total_seconds() / 3600,
                'similarity_threshold': self.similarity_threshold,
                'embeddings_model': index_info.get('embeddings_model', self.embeddings_model),
                'embeddings_provider': index_info.get('embeddings_provider', 'google_generative_ai')
            }
            
        except Exception as e:
            log(f"Error getting index stats: {e}", level='error')
            return {'status': 'error', 'error': str(e)}
    
    async def rebuild_index(self, directory_path: str = None) -> bool:
        """Force rebuild the index with Google embeddings."""
        if directory_path:
            self.data_dir = directory_path
        elif not self.data_dir:
            log("No data directory specified", level='error')
            return False
        
        return await self.build_index(self.data_dir, force_rebuild=True)


# For testing with multiple file support and detailed formatting
async def main():
    """Test function for the Google embeddings migration."""
    # API key is now handled by security_config
    try:
        agent = RAGAgent()  # Will use security_config.google_api_key automatically
    except ValueError as e:
        print(f"Configuration error: {e}")
        print("Please set GOOGLE_API_KEY in your .env file")
        return
    await agent.build_index("C:\\Users\\dhuu3\\Desktop\\local-classify-docs-ai-agent\\data")
    
    # Test with simple format (for compatibility)
    print("=== SIMPLE FORMAT ===")
    result = await agent.invoke("Tìm file có nội dung kế hoạch", "test_session", show_detailed_results=False)
    print(result['content'])
    print()
    
    # Test with detailed format (new feature)
    print("=== DETAILED FORMAT ===")
    result = await agent.invoke("Tìm file có nội dung chứa Doanh thu", "test_session", show_detailed_results=True)
    print(result['content'])
    
    # Show index stats
    print("\n=== INDEX STATS ===")
    stats = agent.get_index_stats()
    print(f"Status: {stats['status']}")
    print(f"Embeddings provider: {stats.get('embeddings_provider', 'unknown')}")
    print(f"Embeddings model: {stats.get('embeddings_model', 'unknown')}")
    print(f"Documents: {stats.get('num_documents', 0)}")
    print(f"Chunks: {stats.get('num_chunks', 0)}")

if __name__ == "__main__":
    asyncio.run(main())