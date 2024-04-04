---
title: "Machine Learning Engineer Internship @ EdPlus at ASU"
excerpt: |
  I had the unique opportunity to intern with EdPlus at ASU’s Artificial Intelligence Product Department (AIPD) team, as part of both my professional development and part of the UA HLT MS program. EdPlus’s AIPD is a small unit at the university that works with cutting edge technologies, and develops various applications and tools to enhance student journeys from start to finish. During my internship, ASU announced its partnership with OpenAI, and I gained expertise in working with Python’s openai library in particular as part of this learning experience.

  The internship allowed me to develop my programming skills in a practical way, where I often had to learn how to accomplish small tasks as part of the bigger project as I moved along. A challenge I faced was having no experience with LLMs or OpenAI in particular at the start of the internship, but after getting to experiment, research, and use this library to produce valuable output, I now feel extremely confident with it. I found using OpenAI’s tools in Python can accomplish many NLP related tasks that I learned about in my program, such as tokenization (their ‘tiktoken’ library offers a free way to tokenize text), information retrieval and question answering (such as setting up RAG to answer questions from specific documents), information extraction and summarization (extracting relevant information from longer texts), creating vector embeddings for text, and more. Where the LLM sometimes fell short, I was able to leverage libraries like nltk that I already had exposure to and experience with through my HLT program.

  Please see the internship page for detailed information about the projects I worked on during my internship.
collection: portfolio
---

During my six month duration of working with EdPlus at ASU’s AIPD team part time, I worked on a total of three main projects, which are discussed individually below. The first project listed, the ‘Course Description Project’, is where I spent the majority of my time, and is thus the most complex project with more content than the other two.

## Course Description Project

_Main Python toolkits used: pandas, openai, re, nltk, pdfminer, tiktoken_

**High level project context:** The university asked for a way to get new course descriptions on a new course page website, and to use various syllabi to base the descriptions off of. They wanted to use AI to generate the descriptions.

My role was to write Python scripts that use openAI’s APIs to generate course descriptions and other course information, based on the provided syllabi. The project was intended as a POC (proof of concept).

**Project status:** POC completed; course descriptions delivered and used on this site: https://courses-dev.asuonline.asu.edu/ 

### In depth project overview & process steps
 
#### 1. Read in syllabi (pdfs only), scrape text from pdfs

I was given a link to a dropbox that contained a mess of syllabi provided by numerous ASU colleges. Most of the syllabi were PDF files, so the first challenge was to find a convenient way to extract the text from the PDFs (which after doing some quick research I found a solution in the Python library pdfminer).

_Extracting pdf snippet:_
```python

#Extract text from PDFS. Re-written to accommodate weird issue with some PDFs not being extracted with extract_pages().
def extract_pdf_text(file_path, token_limit=500):
   print(f"Extracting data from PDF: {file_path}")
   laparams = LAParams(line_margin=1)

   #Use extract_text to get all text from the PDF
   text = extract_text(file_path, laparams=laparams)

   if not text:
    print(f"ERROR: Nothing was extracted from PDF: {file_path}")
        return ""

   #Split the text by whitespace to approximate tokens
   tokens = tokenizer.encode(text) if hasattr(tokenizer, 'encode') else text.split()

   #If the number of tokens exceeds the limit, truncate the tokens
   if len(tokens) > token_limit:
       #Reconstruct text up to the token limit
       truncated_text = ' '.join(tokens[:token_limit])
       return truncated_text
   else:
       #Return the full text if token count is within limits
       return text
```

### 2. Use OpenAI to extract the relevant information from each syllabi

By chunking out each individual lengthy syllabus into smaller texts, I used prompt engineering techniques and an openai chatbot class to ‘extract’ the type of information from each text chunk that we wanted to keep. This text output was saved to a csv for easy access to review it, and also to feed into the second openai chatbot class as described in the next step.

_Text chunking snippet:_
```python
def create_text_chunks(text, chunk_size = 1800):
   chunks = []
   chunk = ""

   # splitting on periods in an attempt to keep sentences together in chunks
   for sentence in text.split('.'):
       if len(tokenizer.encode(chunk)) + len(tokenizer.encode(sentence)) < chunk_size:
           chunk += sentence + '.'
       else:
           chunks.append(chunk)
           chunk = sentence + '.'
   chunks.append(chunk)
   return chunks
```