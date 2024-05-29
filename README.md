# Churry: Organic Chemistry Chatbot

Data is taken from Organic Chemistry: A Tenth Edition by John McMurry which is free to use under CC BY-NC-SA 4.0 license by [OpenStax](https://openstax.org/details/books/organic-chemistry). I named this Churry as a playful abbreviation of Organic Chemistry McMurry.

I am a newbie, so if you find any problems or suggestions about the application, please let me know by mailing me at contact@riaayupramudita.id.

## How to Run
### Clone the repository
Project repo:
```bash
https://github.com/RiaAyuP/churry
```
### Create a conda environment after opening the repository
```bash
conda create -n churry python=3.8 -y
```
```bash
conda activate churry
```

### Install the requirements
```bash
pip install -r requirements.txt
```

### Create a `.env` file in the root directory and add your OpenAI credentials as follows:
```bash
OPENAI_API_KEY = "YOUR OPENAI KEY"
```

### Download the PDF (or any other PDF you want actually) into the folder 'data'
Please download the Organic Chemistry: A Tenth Edition by John McMurry PDF file [here](https://openstax.org/details/books/organic-chemistry)

### Run the embedding application
```bash
python store_index.py
```
Please note that it can take a while if your PDF is large.

### Run the chatbot application
```bash
python app.py
```
It will open in the local host http://127.0.0.1:8080/ and you can start chatting with your PDF!

## Techstack
1. Python
2. LangChain
3. Flask
4. OpenAI
5. ChromaDB

## References
1. [LangChain: Chat with Your Data by Harrison Chase in DeepLearning.AI](https://www.deeplearning.ai/short-courses/langchain-chat-with-your-data/)
2. [End-to-End Medical Chatbot using Llama2 by Boktiar Ahmed Bappy](https://github.com/entbappy/End-to-end-Medical-Chatbot-using-Llama2/tree/main)
3. [LangChain WebScraper Demo by Jason Webster](https://github.com/jasonrobwebster/langchain-webscraper-demo)
