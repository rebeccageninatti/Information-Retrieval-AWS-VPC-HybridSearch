"""
Modulo per il chunking del testo pulito.
Fornisce interfacce per diverse strategie di chunking (RecursiveCharacter, Token-based, Markdown-Header-based).
"""

import logging
from typing import List, Dict, Any
from langchain_text_splitters import RecursiveCharacterTextSplitter, TokenTextSplitter, MarkdownHeaderTextSplitter

logger = logging.getLogger(__name__)

class DocumentChunker:
    """
    Classe responsabile per la divisione dei documenti puliti in frammenti (chunk).
    """

    def __init__(self, separators: List[str] | None = None):
        # Separatori di default se non forniti
        self.separators = separators or ["\n\n", "\n", ". ", " ", ""]
        
        # Definiamo gli header di markdown su cui fare lo splitting di primo livello
        self.headers_to_split_on = [
            ("#", "Header 1"),
            ("##", "Header 2"),
            ("###", "Header 3"),
            ("####", "Header 4"),
        ]
        self.markdown_header_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=self.headers_to_split_on,
            strip_headers=False  # Mantiene gli header nel testo per preservare la leggibilità strutturale
        )

    def split_recursive(self, text: str, chunk_size: int, chunk_overlap: int) -> List[str]:
        """
        Divide il testo in chunk usando RecursiveCharacterTextSplitter.
        
        Args:
            text: Testo di input pulito.
            chunk_size: Dimensione target in caratteri.
            chunk_overlap: Sovrapposizione target in caratteri.
            
        Returns:
            Lista di stringhe (chunk).
        """
        if not text:
            return []
            
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=self.separators,
            keep_separator=True
        )
        return splitter.split_text(text)

    def split_tokens(self, text: str, chunk_size: int, chunk_overlap: int, encoding_name: str = "cl100k_base") -> List[str]:
        """
        Divide il testo in chunk usando TokenTextSplitter.
        
        Args:
            text: Testo di input pulito.
            chunk_size: Dimensione target in token.
            chunk_overlap: Sovrapposizione target in token.
            encoding_name: Nome del tokenizer.
            
        Returns:
            Lista di stringhe (chunk).
        """
        if not text:
            return []

        splitter = TokenTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            encoding_name=encoding_name
        )
        return splitter.split_text(text)

    def split_markdown_two_stage(self, text: str, chunk_size: int, chunk_overlap: int, inject_headers: bool = True) -> List[Dict[str, Any]]:
        """
        Esegue il chunking a due stadi:
        1. Splitta in base agli header Markdown (#, ##, ###, ####) per preservare la coerenza logica.
        2. Per ogni blocco ottenuto, applica il RecursiveCharacterTextSplitter se supera la dimensione massima.
        
        Args:
            text: Testo di input pulito.
            chunk_size: Dimensione target in caratteri del splitter ricorsivo.
            chunk_overlap: Sovrapposizione in caratteri.
            inject_headers: Se True, inietta gli header come stringa di contesto all'inizio di ciascun sotto-chunk.
            
        Returns:
            Lista di dizionari, ognuno contenente il 'text' del chunk e i 'metadata' degli header.
        """
        if not text:
            return []
            
        # Stadio 1: Split basato su Header Markdown
        header_splits = self.markdown_header_splitter.split_text(text)
        
        final_chunks = []
        
        # Inizializza lo splitter ricorsivo per lo Stadio 2
        recursive_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=self.separators,
            keep_separator=True
        )
        
        for doc in header_splits:
            content = doc.page_content
            metadata = doc.metadata
            
            # Costruisce la stringa degli header per l'iniezione (es. "VPC > Route Tables")
            header_path_list = []
            for h_level in ["Header 1", "Header 2", "Header 3", "Header 4"]:
                if h_level in metadata:
                    header_path_list.append(metadata[h_level])
            
            header_context = " > ".join(header_path_list)
            
            # Se la sezione è corta, non c'è bisogno di fare un secondo split
            if len(content) <= chunk_size:
                final_text = content
                if inject_headers and header_context:
                    final_text = f"Context: {header_context}\n\n{content}"
                    
                final_chunks.append({
                    "text": final_text,
                    "metadata": metadata,
                    "header_path": header_context
                })
            else:
                # Stadio 2: Splitta ulteriormente con lo splitter ricorsivo
                sub_splits = recursive_splitter.split_text(content)
                for sub_text in sub_splits:
                    final_text = sub_text
                    if inject_headers and header_context:
                        final_text = f"Context: {header_context}\n\n{sub_text}"
                        
                    final_chunks.append({
                        "text": final_text,
                        "metadata": metadata,
                        "header_path": header_context
                    })
                    
        return final_chunks
