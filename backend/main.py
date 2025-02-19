from typing import Annotated
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
import PyPDF2
from io import BytesIO
import os
from typing import List, Dict
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch
from pdf2image import convert_from_bytes
import pytesseract

app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize CodeLlama
try:
    model_name = "codellama/CodeLlama-7b-Instruct-hf"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        load_in_8bit=True
    )
    
    def generate_summary(text: str) -> str:
        text = text.strip()
        if len(text) > 4000:
            text = text[:4000] + "..."
            
        prompt = f"""<s>[INST] You are a legal document assistant. Please provide a clear and concise summary of the following legal document text, focusing on key points and important details:

{text}

Provide a professional summary in 3-4 sentences.
[/INST]"""
        
        try:
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            outputs = model.generate(
                **inputs,
                max_new_tokens=250,  # Adjusted for longer summaries
                temperature=0.3,
                do_sample=True,
                top_p=0.9,
                top_k=40,
                num_return_sequences=1,
                pad_token_id=tokenizer.eos_token_id,
                repetition_penalty=1.2
            )
            
            summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
            if "[/INST]" in summary:
                summary = summary.split("[/INST]")[1].strip()
            return summary
        except Exception as e:
            print(f"Error generating summary: {str(e)}")
            return "Error generating summary. Please try again."

except Exception as e:
    print(f"Error loading CodeLlama model: {str(e)}")
    model = None
    tokenizer = None
    generate_summary = None

@app.get("/scan-folders")
async def scan_folders():
    """Scan legal folder for PDF files"""
    try:
        # Get the absolute path to the legal directory
        legal_dir = Path(__file__).parent.parent
        pdf_files = []
        
        # Recursively scan all subdirectories
        for pdf_path in legal_dir.rglob("*.pdf"):
            # Get relative path from legal directory
            rel_path = pdf_path.relative_to(legal_dir)
            # Split into directory parts
            path_parts = list(rel_path.parts)
            
            pdf_files.append({
                "path": str(pdf_path),
                "filename": path_parts[-1],
                "folder_path": str(Path(*path_parts[:-1])) if len(path_parts) > 1 else "",
                "folder": path_parts[-2] if len(path_parts) > 1 else "root"
            })
        
        return {
            "files": sorted(pdf_files, key=lambda x: (x["folder_path"], x["filename"])),
            "total": len(pdf_files)
        }
    except Exception as e:
        return {"status": "error", "message": f"Error scanning folders: {str(e)}"}

@app.post("/process-pdf")
async def process_pdf(
    file: Annotated[UploadFile, File(description="PDF file to process")] = None,
    file_path: Annotated[str, Form()] = None
):
    """Process a PDF file and extract context"""
    try:
        if file:
            # Handle uploaded file
            content = await file.read()
            filename = file.filename
            if not filename.lower().endswith('.pdf'):
                return {"status": "error", "message": "File must be a PDF"}
        elif file_path:
            # Handle file from path
            legal_dir = Path(__file__).parent.parent
            pdf_path = legal_dir / file_path
            
            # Validate path is within legal directory
            if not pdf_path.resolve().is_relative_to(legal_dir.resolve()):
                return {"status": "error", "message": "Invalid file path"}
                
            if not pdf_path.exists():
                return {"status": "error", "message": "File not found"}
                
            if not pdf_path.suffix.lower() == '.pdf':
                return {"status": "error", "message": "File must be a PDF"}
                
            with open(pdf_path, 'rb') as f:
                content = f.read()
            filename = pdf_path.name
        else:
            return {"status": "error", "message": "No file provided"}
            
        if not content:
            return {"status": "error", "message": "Empty file"}
        
        # Try to extract text using PyPDF2 first
        text = ""
        try:
            pdf_reader = PyPDF2.PdfReader(BytesIO(content))
            if len(pdf_reader.pages) == 0:
                return {"status": "error", "message": "PDF file has no pages"}
            
            # Extract text from all pages
            for page in pdf_reader.pages:
                text += page.extract_text()
        except Exception as e:
            print(f"PyPDF2 extraction failed: {str(e)}")
            
        # If no text found, try OCR
        if not text.strip():
            try:
                # Convert PDF to images
                images = convert_from_bytes(content)
                
                # Extract text from each image using OCR
                for image in images:
                    text += pytesseract.image_to_string(image) + "\n"
                    
                if not text.strip():
                    return {"status": "error", "message": "Could not extract text from PDF (tried both text extraction and OCR)"}
            except Exception as e:
                print(f"OCR extraction failed: {str(e)}")
                return {"status": "error", "message": f"Failed to process PDF: {str(e)}"}
        
        # Generate summary
        try:
            if generate_summary:
                # Split text into chunks if it's too long
                max_chunk_length = 2048
                chunks = [text[i:i + max_chunk_length] for i in range(0, len(text), max_chunk_length)]
                
                # Summarize each chunk
                summaries = []
                for chunk in chunks[:3]:
                    if len(chunk.strip()) > 100:
                        summary = generate_summary(chunk)
                        summaries.append(summary)
                
                final_summary = " ".join(summaries)
            else:
                final_summary = "CodeLlama model not available. Showing text preview only."
        except Exception as e:
            print(f"Summarization error: {str(e)}")
            final_summary = "Error generating summary. Showing text preview only."
        
        return {
            "text": text[:1000],  # Return first 1000 chars as preview
            "summary": final_summary,
            "status": "success",
            "filename": filename,
            "total_pages": len(pdf_reader.pages) if 'pdf_reader' in locals() else len(images)
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.post("/chat")
async def chat(
    message: Annotated[str, Form()],
    context: Annotated[str, Form()]
):
    """Chat endpoint that uses context for responses"""
    try:
        if not model or not tokenizer:
            return {"status": "error", "message": "CodeLlama model not available"}
            
        prompt = f"""<s>[INST] Using the following context from a legal document, answer the question.

Context: {context}

Question: {message}

Answer: [/INST]"""
        
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        outputs = model.generate(
            **inputs,
            max_length=50000,
            temperature=0.1,
            do_sample=True,
            top_p=0.95,
            top_k=50,
            num_return_sequences=1
        )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return {"response": response}
    except Exception as e:
        return {"status": "error", "message": str(e)} 