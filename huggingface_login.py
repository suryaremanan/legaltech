"""
Log in to Hugging Face to enable model downloads.
"""
import sys
import getpass
from huggingface_hub import login

def main():
    """Log in to Hugging Face Hub with user credentials."""
    print("This script will help you log in to Hugging Face.")
    print("You'll need a Hugging Face account (signup at https://huggingface.co/join).")
    print("\nIf you don't want to log in, you can run the PDF converter with the --offline flag:")
    print("python pdf_to_jsonl_converter.py --offline\n")
    
    token = getpass.getpass("Enter your Hugging Face token (from https://huggingface.co/settings/tokens): ")
    if not token:
        print("No token provided. Exiting.")
        return
        
    try:
        login(token="")
        print("\n✅ Successfully logged in to Hugging Face!")
        print("You can now run the PDF converter without the --offline flag.")
    except Exception as e:
        print(f"\n❌ Failed to log in: {e}")
        print("You can still run the converter in offline mode:")
        print("python pdf_to_jsonl_converter.py --offline")

if __name__ == "__main__":
    main() 
