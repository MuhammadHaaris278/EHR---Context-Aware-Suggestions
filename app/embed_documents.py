"""
Document embedding script for clinical knowledge base.
Processes and embeds medical documents into FAISS index.
"""

import os
import asyncio
import argparse
from pathlib import Path
from typing import List, Dict, Optional
import logging
from langchain.schema import Document
from langchain.document_loaders import (
    TextLoader, PDFPlumberLoader, UnstructuredWordDocumentLoader
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
import json

from .retriever import ClinicalRetriever

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DocumentEmbedder:
    """
    Document embedding utility for clinical knowledge base.
    Handles loading, processing, and embedding of medical documents.
    """
    
    def __init__(self, index_path: str = "faiss_index"):
        self.index_path = index_path
        self.retriever = ClinicalRetriever(index_path=index_path)
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        
    async def initialize(self):
        """Initialize the retriever."""
        await self.retriever.initialize()
        logger.info("Document embedder initialized")
    
    async def embed_directory(self, directory_path: str, file_pattern: str = "*"):
        """
        Embed all documents in a directory.
        
        Args:
            directory_path: Path to directory containing documents
            file_pattern: File pattern to match (e.g., "*.pdf", "*.txt")
        """
        try:
            directory = Path(directory_path)
            if not directory.exists():
                raise FileNotFoundError(f"Directory not found: {directory_path}")
            
            # Find all matching files
            files = list(directory.glob(file_pattern))
            logger.info(f"Found {len(files)} files to process")
            
            # Process each file
            for file_path in files:
                try:
                    await self.embed_file(str(file_path))
                except Exception as e:
                    logger.error(f"Error processing {file_path}: {e}")
                    continue
            
            logger.info(f"Completed embedding {len(files)} files")
            
        except Exception as e:
            logger.error(f"Error embedding directory: {e}")
            raise
    
    async def embed_file(self, file_path: str):
        """
        Embed a single document file.
        
        Args:
            file_path: Path to the document file
        """
        try:
            logger.info(f"Processing file: {file_path}")
            
            # Load document
            documents = self._load_document(file_path)
            
            if not documents:
                logger.warning(f"No content loaded from {file_path}")
                return
            
            # Add metadata
            for doc in documents:
                doc.metadata.update({
                    "source": file_path,
                    "filename": os.path.basename(file_path),
                    "type": "clinical_document"
                })
            
            # Add to retriever
            await self.retriever.add_documents(documents)
            
            logger.info(f"Successfully embedded {len(documents)} chunks from {file_path}")
            
        except Exception as e:
            logger.error(f"Error embedding file {file_path}: {e}")
            raise
    
    def _load_document(self, file_path: str) -> List[Document]:
        """Load document based on file type."""
        file_extension = Path(file_path).suffix.lower()
        
        try:
            if file_extension == '.txt':
                loader = TextLoader(file_path, encoding='utf-8')
            elif file_extension == '.pdf':
                loader = PDFPlumberLoader(file_path)
            elif file_extension in ['.doc', '.docx']:
                loader = UnstructuredWordDocumentLoader(file_path)
            else:
                logger.warning(f"Unsupported file type: {file_extension}")
                return []
            
            # Load and split documents
            documents = loader.load()
            split_documents = self.text_splitter.split_documents(documents)
            
            return split_documents
            
        except Exception as e:
            logger.error(f"Error loading document {file_path}: {e}")
            return []
    
    async def embed_text_content(self, content: str, metadata: Dict):
        """
        Embed raw text content.
        
        Args:
            content: Raw text content
            metadata: Document metadata
        """
        try:
            document = Document(page_content=content, metadata=metadata)
            documents = self.text_splitter.split_documents([document])
            
            await self.retriever.add_documents(documents)
            
            logger.info(f"Embedded {len(documents)} chunks from text content")
            
        except Exception as e:
            logger.error(f"Error embedding text content: {e}")
            raise
    
    async def embed_clinical_guidelines(self, guidelines_file: str):
        """
        Embed clinical guidelines from a structured JSON file.
        
        Args:
            guidelines_file: Path to JSON file containing clinical guidelines
        """
        try:
            with open(guidelines_file, 'r', encoding='utf-8') as f:
                guidelines_data = json.load(f)
            
            documents = []
            
            for guideline in guidelines_data.get('guidelines', []):
                content = self._format_guideline_content(guideline)
                
                doc = Document(
                    page_content=content,
                    metadata={
                        "source": guidelines_file,
                        "type": "clinical_guideline",
                        "title": guideline.get("title", "Unknown"),
                        "specialty": guideline.get("specialty", "General"),
                        "condition": guideline.get("condition", "Unknown")
                    }
                )
                
                documents.append(doc)
            
            # Split and embed
            split_documents = self.text_splitter.split_documents(documents)
            await self.retriever.add_documents(split_documents)
            
            logger.info(f"Embedded {len(split_documents)} guideline chunks")
            
        except Exception as e:
            logger.error(f"Error embedding clinical guidelines: {e}")
            raise
    
    def _format_guideline_content(self, guideline: Dict) -> str:
        """Format guideline data into readable text."""
        content_parts = []
        
        if guideline.get("title"):
            content_parts.append(f"Title: {guideline['title']}")
        
        if guideline.get("condition"):
            content_parts.append(f"Condition: {guideline['condition']}")
        
        if guideline.get("summary"):
            content_parts.append(f"Summary: {guideline['summary']}")
        
        if guideline.get("recommendations"):
            content_parts.append("Recommendations:")
            for i, rec in enumerate(guideline["recommendations"], 1):
                content_parts.append(f"{i}. {rec}")
        
        if guideline.get("contraindications"):
            content_parts.append("Contraindications:")
            for contraindication in guideline["contraindications"]:
                content_parts.append(f"- {contraindication}")
        
        if guideline.get("monitoring"):
            content_parts.append(f"Monitoring: {guideline['monitoring']}")
        
        return "\n\n".join(content_parts)
    
    async def create_sample_knowledge_base(self):
        """Create a sample knowledge base with common clinical scenarios."""
        
        sample_documents = [
            {
                "content": """
                Chest Pain Assessment Guidelines
                
                For patients presenting with chest pain:
                
                1. Immediate assessment of vital signs and oxygen saturation
                2. Obtain 12-lead ECG within 10 minutes of arrival
                3. Assess for signs of acute coronary syndrome
                4. Consider cardiac enzymes (troponin) if cardiac cause suspected
                5. Chest X-ray to evaluate for pulmonary causes
                6. Risk stratification using HEART score or similar
                7. Consider stress testing for intermediate risk patients
                8. Discharge planning with appropriate follow-up
                
                Red flags requiring immediate cardiology consultation:
                - ST-elevation on ECG
                - Elevated troponin levels
                - Hemodynamic instability
                - New heart murmur
                """,
                "metadata": {
                    "source": "sample_guidelines",
                    "type": "clinical_guideline",
                    "title": "Chest Pain Assessment",
                    "specialty": "Emergency Medicine",
                    "condition": "Chest Pain"
                }
            },
            {
                "content": """
                Abdominal Pain Evaluation
                
                Systematic approach to abdominal pain:
                
                1. Complete history including pain characteristics, location, timing
                2. Physical examination including vital signs
                3. Laboratory studies: CBC, CMP, lipase, urinalysis
                4. Imaging studies based on clinical suspicion
                5. Consider appendicitis, cholecystitis, bowel obstruction
                6. Special considerations for elderly patients
                7. Gynecologic causes in women of childbearing age
                8. Surgical consultation for peritoneal signs
                
                Warning signs requiring immediate intervention:
                - Peritoneal signs
                - Hemodynamic instability
                - Severe pain out of proportion to exam
                - Signs of bowel obstruction
                """,
                "metadata": {
                    "source": "sample_guidelines",
                    "type": "clinical_guideline",
                    "title": "Abdominal Pain Evaluation",
                    "specialty": "Emergency Medicine",
                    "condition": "Abdominal Pain"
                }
            },
            {
                "content": """
                Diabetes Management Guidelines
                
                Routine diabetes care recommendations:
                
                1. HbA1c monitoring every 3-6 months
                2. Annual comprehensive foot examination
                3. Annual dilated eye examination
                4. Annual nephropathy screening
                5. Blood pressure monitoring and control
                6. Lipid screening and management
                7. Immunizations (influenza, pneumococcal)
                8. Medication adherence assessment
                9. Lifestyle counseling
                10. Diabetes self-management education
                
                Target goals:
                - HbA1c < 7% for most adults
                - Blood pressure < 130/80 mmHg
                - LDL cholesterol < 100 mg/dL
                """,
                "metadata": {
                    "source": "sample_guidelines",
                    "type": "clinical_guideline",
                    "title": "Diabetes Management",
                    "specialty": "Endocrinology",
                    "condition": "Diabetes Mellitus"
                }
            }
        ]
        
        for doc_data in sample_documents:
            await self.embed_text_content(
                doc_data["content"],
                doc_data["metadata"]
            )
        
        logger.info("Sample knowledge base created successfully")

async def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(description="Embed clinical documents into FAISS index")
    parser.add_argument("--directory", "-d", help="Directory containing documents to embed")
    parser.add_argument("--file", "-f", help="Single file to embed")
    parser.add_argument("--pattern", "-p", default="*", help="File pattern to match")
    parser.add_argument("--index", "-i", default="faiss_index", help="FAISS index path")
    parser.add_argument("--sample", "-s", action="store_true", help="Create sample knowledge base")
    
    args = parser.parse_args()
    
    # Initialize embedder
    embedder = DocumentEmbedder(index_path=args.index)
    await embedder.initialize()
    
    try:
        if args.sample:
            logger.info("Creating sample knowledge base...")
            await embedder.create_sample_knowledge_base()
        
        if args.directory:
            logger.info(f"Embedding documents from directory: {args.directory}")
            await embedder.embed_directory(args.directory, args.pattern)
        
        if args.file:
            logger.info(f"Embedding single file: {args.file}")
            await embedder.embed_file(args.file)
        
        if not any([args.directory, args.file, args.sample]):
            logger.info("No input specified. Creating sample knowledge base...")
            await embedder.create_sample_knowledge_base()
        
        # Print stats
        stats = await embedder.retriever.get_stats()
        logger.info(f"Index statistics: {stats}")
        
    except Exception as e:
        logger.error(f"Error in main process: {e}")
        raise

if __name__ == "__main__":
    asyncio.run(main())