---
title: "Retrieval-Augmented Generation (RAG) for EdPlus Guru Data Project"
excerpt: |
  This was a project I worked on as part of my internship at EdPlus at ASU. The task in this project was to set up a RAG system to handle ASU Online data that lives in Guru that the Success and Enrollment Coaches use, so that they could ask questions and get answers. The main NLP tasks this project relates to is Question/Answering and information retrieval, and I was able to use Python's openai library to accomplish much of it.

collection: portfolio
---


## Create Retrieval-Augmented Generation (RAG) for EdPlus Guru Data Project

_Main Python toolkits used: pandas, openai, tiktoken, re, BeautifulSoup_

**High level project context:** EdPlus employees at the call center often get asked questions by students, that they have to dig through many pages of data on a Guru repository to find answers to. The university requested that we use AI to enable the employees to find answers to these questions when they come up, where the answers are based on actual university data. Additionally, they requested that the eventual application should also suggest related questions in case those were helpful to the student.

My role was mainly to upskill and learn about RAG. I wrote Python scripts that used ASU Guru data to set up a RAG system that could help call center employees get answers to questions students have, that currently lives in Guru. My involvement in this project was intended to get a larger scale project started as a _very_ small POC so I could upskill, before the project got transferred as part of a larger university initative that involved additional development teams outside of EdPlus to take the lead on (while still working with the lead developer(s) on the AIPD team).

**Project status:** Mini POC (proof of concept) completed; Example output was available in a Python notebook that I handed off to the team. This project did not have a real front end at the time I finished working on it.

### In depth project overview & process steps
 
#### 1. Read in JSON data from Guru cards & clean data

My coworker provided me with a compressed folder with JSON files he’d created from the data on Guru. The content he extracted from each page was still in html format, so the first step was to remove the html tags and do some general clean up on the text. I used python’s BeautifulSoup library to accomplish this. I tried to preserve some formatting such as trying to make it clear when any data was in a table format. 

_Snippet of part of the data cleaning script:_
```python
#Function to remove html tags from the 'content' in the JSON data
def remove_html_tags(html_content):
   soup = BeautifulSoup(html_content, 'lxml')

   #Links
   for a in soup.find_all('a'):
       link_text = a.get_text(strip=True)
       url = a.get('href', '')
       if link_text and url and link_text != url:
           a.replace_with(f' {link_text} ({url}) ')
       else:
           a.replace_with(f' {url} ')

   #Lists
   for ul in soup.find_all(['ul', 'ol']):
       for li in ul.find_all('li'):
           li.replace_with(f' - {li.get_text(strip=True)}')
       ul.replace_with('' + ul.get_text())

   #Headers
   for header in soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6']):
       content = ['' + header.get_text(strip=True) + '']
       next_node = header.find_next_sibling()
       while next_node and not next_node.name in ['h1', 'h2', 'h3', 'h4', 'h5', 'h6']:
           if next_node.name in ['p', 'div', 'details']:
               text = next_node.get_text(separator='', strip=True)
               if text:
                   content.append(text)
           next_node = next_node.find_next_sibling()

       #header.replace_with(''.join(content))

   #Extract text
   text_only = soup.get_text(separator='\n', strip=True)
   return text_only
```

#### 2. Generate vector embeddings (using OpenAI) for the cleaned up text

Once I was happy with the cleaned up content text, I used that text to generate vector embeddings using OpenAI. The vector embeddings were added to a pandas df with the other data from the JSON objects.

_Code snippet:_
```python
#Function to get openAI embeddings
def get_embeddings(text):
   response = openai.Embedding.create(
       model = "text-embedding-ada-002",
       input = [text]
   )
   embedding = response["data"][0]["embedding"]
   return embedding

#Function to generate embeddings for all items in the data
def generate_embeddings(all_data):
   for index, item in enumerate(all_data):
       try:
           embedding = get_embeddings(item['all_content'])
           item['embedding'] = embedding
       except Exception as e:
          #Limiting content display to first 100 characters
           print(f"Error occurred at index {index} with content: {item['all_content'][:100]}")  
           print("Error details:", e)
           error_list.append(index)
```

#### 3. Set up RAG for the cleaned up texts & their vector embeddings

This was the most crucial step for this project, and was a technique I had not learned about in my HLT program (but it’s extremely useful and quite relevant in the industry today). RAG allows for an LLM that was not trained on specific data to add that data to the LLM, so that it can answer questions about something catered to a specific company (or whatever data you provide it). Learning about RAG was the biggest reason I was instructed to take this project on. The code works by generating vector embeddings for any query, and then grabbing the text chunks that are closest to it based on the cosine similarity between that query’s embedding and the embeddings it generated for each text (in this case the Guru text data). In this way the AI can provide tailored responses based on information it would otherwise not ‘know’, since it wasn’t trained on that data.

You can view the chatbot used for this step on my [Chatbot Samples GitHub Repository](https://github.com/kelynnski/chatbot-samples/blob/main/rag_chatbot.py).

#### 4. Send some questions to the AI (any question) for testing

The next step was to write up some example questions to show the output. In addition to answering the question, I also programmed the small app to send back three related questions for every question the user asked (code snippet is in the link above, in the function ‘generate_questions()’). The output was pretty accurate. This project did not end up getting used in any real ASU application as of this point, but it was an excellent learning experience where I gained insights into data cleaning and got to use openai’s vector embedding generation (and of course, RAG in general).

_Some example snippets of the output with questions in a Python notebook:_
```python
employee.answer_question("What types of careers can I have with an engineering degree?", debug=True)
try:
   related_questions, history = employee.generate_questions()
   print("Related Questions:", related_questions)
   print("Conversation History:", history)
except Exception as e:
   print("An error occurred while generating questions:", e)
```
>Output: With an engineering degree, you can pursue various career paths depending on your specialization. Some potential careers include application developer, computer programmer, software architect, software developer, computer system analyst, director of engineering, engineering administrator, engineering project manager, systems and software manager, strategic technology developer, and more. These careers offer opportunities in industries such as software engineering, technology management, and sustainable engineering, among others.<br>
>Related questions generated: <br>
>1. What are some specific job titles that someone with an engineering degree can pursue?
>2. Are there any industries or sectors that commonly hire individuals with an engineering degree?
>3. Can you provide examples of how an engineering degree can lead to a variety of career paths?

#### 5. Test the set up

One of the challenges with RAG is figuring out how best to see how well it is performing. A solution I used was to prepare specific validation questions with the best matches manually identified, and see if it was using those matches as the source. Out of 5 example questions, 4 of the 5 included the best match in the top two closest results.

_Some example validation questions and answers:_
```python
validation_qs = ["Does the Sociology BS program have accreditation?", "Who is the accrediting agency for the Leadership & Management, MLM program?",
               "Describe the experiential learning opportunities (clinicals, internship, practicum, research, student teaching, etc. for the Complex Systems Science, MS program.",
                "What kind of work is an ideal candidate doing for the Social Entrepreneurship & Community Development Graduate Certificate?",
               "Is the Biomimicry Graduate Certificate program eligible for course recommendations and quick enrolls?"]
validation_cards = ['card_2143', 'card_48', 'card_832', 'card_680', 'card_0']
validation_urls = ['https://app.getguru.com/card/TodBgxBc/Accreditation-Internships-Licensure-Sociology-BS',
                 'https://app.getguru.com/card/cEgK6rgi/Accreditation-Internships-Licensure-State-Restrictions-Leadership-Management-MLM',
                 'https://app.getguru.com/card/iX4RzqqT/Accreditation-Internships-Licensure-Complex-Systems-Science-MS',
                 'https://app.getguru.com/card/i7ppGnoT/Program-Overview-Social-Entrepreneurship-Community-Development-Graduate-Certificate-',
                 'https://app.getguru.com/card/c46oeari/Course-Information-Biomimicry-Graduate-Certificate']

count = 0
for question, url in zip(validation_qs, validation_urls):
   answer, sources = employee.answer_question(question, debug=False)
   if len(sources) >= 2:
       if sources[0][0] == url or sources[1][0] == url:
           print("Pass")
       else:
           print("Fail")
   print(answer, sources)

```
>Output (Pass/Fail status and answer to the questions):<br>
>Pass - No, the Sociology BS program does not have accreditation.<br>
>Fail - The accrediting agency for the Leadership & Management, MLM program is AACSB (Association to Advance Collegiate Schools of Business).<br>
>Pass - Experiential learning opportunities for the Complex Systems Science, MS program include the option for students to engage in research or related activities, typically in the form of a capstone project, under the guidance of a faculty mentor. These opportunities allow students to apply their knowledge and skills in practical settings and gain hands-on experience in complex systems science.<br>
>Pass - An ideal candidate for the Social Entrepreneurship & Community Development Graduate Certificate is an individual who is currently working in or interested in working in any sort of community-facing role, seeking to be innovative in the creation or further development of programs.<br>
>Pass - Yes, the Biomimicry Graduate Certificate program is eligible for course recommendations and quick enrolls. 