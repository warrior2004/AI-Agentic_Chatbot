import os
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)

project_name = "AI_Agentic_Chatbot"

list_of_files = [
    f"src/__init__.py",
    f"src/ai_agent.py",
    f"src/frontend.py",
    f"src/backend.py",
    f"src/logger.py",
    f"src/exception.py",
    f"setup.py",
    "Dockerfile",
    ".dockerignore",
    "requirements.txt",
    "README.md"
]

for filepath in list_of_files:
    filepath = Path(filepath)
    filedir, filename = os.path.split(filepath)

    if filedir != "":
        os.makedirs(filedir, exist_ok=True)
        logging.info(f"creating directory:{filedir} for the file {filename}")
    
    if (not os.path.exists(filepath)) or (os.path.getsize(filepath) == 0):
        with open(filepath,'w') as f:
            pass
        logging.info(f"Creating empty file:{filepath}")

    else:
        logging.info(f"{filename} is already exists")