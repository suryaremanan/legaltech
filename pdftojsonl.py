import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

try:
    import fitz  # PyMuPDF
except ImportError:
    try:
        import PyMuPDF as fitz
        print("Using PyMuPDF import instead of fitz")
    except ImportError:
        print("ERROR: Cannot import fitz or PyMuPDF. Please run fix_pymupdf.py")
        import sys
        sys.exit(1)
import pytesseract
from PIL import Image
import numpy as np
from sentence_transformers import SentenceTransformer
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("pdf_processing.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Updated configuration parameters with fallback models
CONFIG = {
    "legal_dir": "./clients/",  # Base directory containing client folders
    "output_dir": "./processed_data/",  # Directory to store processed JSONL files
    "client_folders": ["@Macromex_JBS_redacted", "@Tazz_Guerrilla_redacted"],  # Client folders to process
    "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",  # Primary model for embeddings
    "fallback_embedding_models": [  # Fallback public models that don't require auth
        "all-MiniLM-L6-v2",
        "paraphrase-MiniLM-L6-v2",
        "distiluse-base-multilingual-cased-v1"
    ],
    "summary_model": "mistralai/Mistral-7B-Instruct-v0.2",  # Model for summarization
    "fallback_summary": True,  # Whether to use rule-based summarization as fallback
    "max_summary_length": 150,  # Maximum length for document summaries
    "ocr_lang": "eng",  # Language for OCR
    "chunk_size": 1000,  # Number of characters per chunk for processing
    "chunk_overlap": 200,  # Overlap between chunks
    "offline_mode": False,  # Set to True to avoid HF model downloads
}

class PDFProcessor:
    def __init__(self, config: Dict):
        self.config = config
        self.embedding_model = self._load_embedding_model()
        self.tokenizer = self._load_tokenizer()
        
        # Only load LLM if GPU is available and not in offline mode
        if torch.cuda.is_available() and not config.get("offline_mode", False):
            try:
                self.summary_model = AutoModelForCausalLM.from_pretrained(config["summary_model"])
                logger.info(f"Loaded summary model: {config['summary_model']}")
            except Exception as e:
                logger.warning(f"Could not load summary model: {e}")
                self.summary_model = None
        else:
            logger.warning("GPU not available or offline mode enabled. Using rule-based summarization.")
            self.summary_model = None
        
        # Create output directory if it doesn't exist
        os.makedirs(config["output_dir"], exist_ok=True)

    def _load_embedding_model(self):
        """Try to load embedding model with fallbacks for authentication issues"""
        # First try the primary model
        try:
            logger.info(f"Attempting to load embedding model: {self.config['embedding_model']}")
            return SentenceTransformer(self.config["embedding_model"])
        except Exception as e:
            logger.warning(f"Failed to load primary embedding model: {e}")
            
            # Try fallback models
            for fallback_model in self.config.get("fallback_embedding_models", []):
                try:
                    logger.info(f"Attempting to load fallback model: {fallback_model}")
                    return SentenceTransformer(fallback_model)
                except Exception as fallback_e:
                    logger.warning(f"Failed to load fallback model {fallback_model}: {fallback_e}")
            
            # Last resort: create a simple embedding function
            logger.warning("All embedding models failed. Using a simple fallback embedding function.")
            
            class SimpleEmbedder:
                def encode(self, texts, **kwargs):
                    """Simple fallback that creates random embeddings"""
                    if isinstance(texts, str):
                        texts = [texts]
                    return np.random.rand(len(texts), 384)  # Create random embeddings of correct dimension
                        
            return SimpleEmbedder()
    
    def _load_tokenizer(self):
        """Load tokenizer with fallback for offline mode"""
        if self.config.get("offline_mode", False):
            logger.info("Offline mode enabled. Using default tokenizer.")
            from transformers import AutoTokenizer
            return AutoTokenizer.from_pretrained("gpt2")  # Most widely available fallback
            
        try:
            return AutoTokenizer.from_pretrained(self.config["summary_model"])
        except Exception as e:
            logger.warning(f"Failed to load tokenizer: {e}. Using default tokenizer.")
            from transformers import AutoTokenizer
            return AutoTokenizer.from_pretrained("gpt2")  # Fallback

    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extract text from PDF files with OCR fallback for images."""
        logger.info(f"Extracting text from {pdf_path}")
        
        try:
            document = fitz.open(pdf_path)
            text = ""
            
            for page_num in range(len(document)):
                page = document[page_num]
                page_text = page.get_text()
                
                # If page has very little text, it might be an image-based page
                if len(page_text.strip()) < 50:
                    logger.info(f"Page {page_num} appears to be image-based, applying OCR")
                    # Convert page to an image
                    pix = page.get_pixmap()
                    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                    # Apply OCR
                    ocr_text = pytesseract.image_to_string(img, lang=self.config["ocr_lang"])
                    text += ocr_text
                else:
                    text += page_text
                
                text += "\n\n"  # Add separation between pages
            
            document.close()
            return text
        except Exception as e:
            logger.error(f"Error processing {pdf_path}: {str(e)}")
            return ""

    def clean_text(self, text: str) -> str:
        """Clean and normalize extracted text."""
        # Remove excessive whitespace
        text = " ".join(text.split())
        # Replace multiple newlines with a single one
        text = "\n".join([line.strip() for line in text.split("\n") if line.strip()])
        return text

    def chunk_text(self, text: str) -> List[str]:
        """Split text into manageable chunks for processing."""
        chunks = []
        start = 0
        text_length = len(text)
        
        while start < text_length:
            end = min(start + self.config["chunk_size"], text_length)
            
            # If we're not at the end, try to find a good breaking point
            if end < text_length:
                # Try to break at paragraph or sentence
                for break_char in ["\n\n", "\n", ". ", "! ", "? "]:
                    pos = text.rfind(break_char, start, end)
                    if pos != -1:
                        end = pos + len(break_char)
                        break
            
            chunks.append(text[start:end])
            start = end - self.config["chunk_overlap"]  # Create overlap between chunks
        
        return chunks

    def generate_summary(self, text: str) -> str:
        """Generate a concise summary of the document with better fallback."""
        if len(text) < 200:  # If text is already short
            return text
            
        try:
            if self.summary_model and not self.config.get("fallback_summary", False):
                # Use the LLM for summarization
                prompt = f"Summarize the following legal document in 3-4 sentences:\n\n{text[:4000]}..."
                inputs = self.tokenizer(prompt, return_tensors="pt")
                
                if torch.cuda.is_available():
                    inputs = inputs.to("cuda")
                
                with torch.no_grad():
                    outputs = self.summary_model.generate(
                        **inputs,
                        max_length=self.config["max_summary_length"],
                        num_return_sequences=1,
                        temperature=0.7
                    )
                
                summary = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                # Extract just the summary part (remove the prompt)
                summary = summary.split(":\n\n")[1] if ":\n\n" in summary else summary
                return summary
            else:
                # Improved rule-based summarization
                return self._rule_based_summary(text)
                
        except Exception as e:
            logger.error(f"Error generating summary: {str(e)}")
            return self._rule_based_summary(text)
    
    def _rule_based_summary(self, text: str) -> str:
        """Generate a simple rule-based summary when LLM is not available."""
        # Get first paragraph
        paragraphs = text.split('\n\n')
        first_para = paragraphs[0] if paragraphs else text[:500]
        
        # Extract key sentences
        sentences = []
        for sent_end in ['. ', '! ', '? ', '\n']:
            sentences.extend(first_para.split(sent_end))
        
        # Filter out very short sentences
        sentences = [s.strip() + '.' for s in sentences if len(s.strip()) > 20]
        
        # Return first few sentences
        return ' '.join(sentences[:3])

    def process_pdf(self, pdf_path: str) -> Tuple[str, str, Dict]:
        """Process a single PDF file and return the text, summary, and metadata."""
        file_name = os.path.basename(pdf_path)
        logger.info(f"Processing {file_name}")
        
        # Extract text from PDF
        raw_text = self.extract_text_from_pdf(pdf_path)
        
        # Clean the text
        cleaned_text = self.clean_text(raw_text)
        
        # Generate summary
        summary = self.generate_summary(cleaned_text)
        
        # Create metadata
        metadata = {
            "filename": file_name,
            "path": pdf_path,
            "char_count": len(cleaned_text),
            "word_count": len(cleaned_text.split()),
        }
        
        return cleaned_text, summary, metadata

    def create_jsonl_entries(self, client_name: str, pdf_files: List[str]) -> List[Dict]:
        """Create JSONL entries for all PDFs in a client folder."""
        entries = []
        
        for pdf_path in pdf_files:
            text, summary, metadata = self.process_pdf(pdf_path)
            
            # Skip empty or very short documents
            if len(text) < 100:
                logger.warning(f"Skipping {pdf_path} due to insufficient content")
                continue
                
            # Create entry for fine-tuning
            entry = {
                "client": client_name,
                "text": text,
                "summary": summary,
                "metadata": metadata,
                # Format specifically for Hugging Face fine-tuning
                "prompt": f"Provide legal analysis based on the following document: {text[:500]}...",
                "completion": f"Based on the document provided, here is a legal analysis: {summary}"
            }
            
            entries.append(entry)
            
        return entries

    def save_jsonl(self, entries: List[Dict], output_path: str) -> None:
        """Save entries to JSONL file."""
        with open(output_path, 'w', encoding='utf-8') as f:
            for entry in entries:
                f.write(json.dumps(entry, ensure_ascii=False) + '\n')
        
        logger.info(f"Saved {len(entries)} entries to {output_path}")

    def process_client_folder(self, client_folder: str) -> None:
        """Process all PDFs in a client folder."""
        client_path = os.path.join(self.config["legal_dir"], client_folder)
        client_name = client_folder.replace("@", "").replace("_redacted", "")
        
        if not os.path.exists(client_path):
            logger.error(f"Client folder {client_path} does not exist")
            return
            
        # Find all PDF files in the client folder (including subdirectories)
        pdf_files = []
        for root, _, files in os.walk(client_path):
            for file in files:
                if file.lower().endswith('.pdf'):
                    pdf_files.append(os.path.join(root, file))
                    
        if not pdf_files:
            logger.warning(f"No PDF files found in {client_path}")
            return
            
        logger.info(f"Found {len(pdf_files)} PDF files in {client_path}")
        
        # Process PDFs and create JSONL entries
        entries = self.create_jsonl_entries(client_name, pdf_files)
        
        # Save to JSONL file
        output_path = os.path.join(self.config["output_dir"], f"{client_name}_data.jsonl")
        self.save_jsonl(entries, output_path)
        
        # Create a merged training file for Hugging Face format
        hf_entries = []
        for entry in entries:
            hf_entry = {
                "text": f"<|prompt|>{entry['prompt']}<|endofprompt|><|answer|>{entry['completion']}<|endofanswer|>"
            }
            hf_entries.append(hf_entry)
            
        hf_output_path = os.path.join(self.config["output_dir"], f"{client_name}_hf_training.jsonl")
        self.save_jsonl(hf_entries, hf_output_path)

    def process_all_clients(self) -> None:
        """Process all client folders."""
        for client_folder in self.config["client_folders"]:
            logger.info(f"Processing client folder: {client_folder}")
            self.process_client_folder(client_folder)


def main():
    """Main function to run the PDF processing pipeline."""
    logger.info("Starting PDF to JSONL conversion process")
    
    try:
        # Check for --offline flag in command line arguments
        import sys
        if "--offline" in sys.argv:
            CONFIG["offline_mode"] = True
            logger.info("Running in offline mode - will use cached models or fallbacks")
        
        processor = PDFProcessor(CONFIG)
        processor.process_all_clients()
        logger.info("PDF to JSONL conversion completed successfully")
    except Exception as e:
        logger.error(f"Error in main processing pipeline: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
    main() 
