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

#### 2. Use OpenAI to extract the relevant information from each syllabi

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

_Extract relevant content snippet (note: content\_extraction\_bot.chatbot() is the chatbot defined to extract the relevant content):_
```python
#Function to extract relevant content from a syllabus.
async def extract_relevant_content(text):
   ai_content_bot = content_extraction_bot.chatbot()
   ai_content_bot.temperature = 1

   prompt = "Extract the relevant information from the syllabus content. Do not include class-specific information such as due dates, the instructor, etc. Also, do not include non-course information such as the university's mission statement and disability resources.\n"
   prompt += "\n### Syllabus Content ###\n"
   prompt += text
   prompt += "\n### End Syllabus Content ###\n"

   (response, input_tokens, output_tokens) = await ai_content_bot.answer_question(prompt)

   input_cost = input_tokens / 1000 * INPUT_COST_PER_1K_TOKENS
   output_cost = output_tokens / 1000 * OUTPUT_COST_PER_1K_TOKENS
   total_cost = input_cost + output_cost
   return (response, total_cost)
```

#### 3. For each syllabi, using OpenAI, generate a course description & other important course information (course code, course title, credit hours, learning outcomes) based on the extracted relevant information

Now that I had much more manageable texts to feed to the AI (since there is a limit on how many tokens can be put into one call and LLMs are also not as effective with super lengthy texts), I again used prompt engineering techniques with openai–this time to have the AI generate data points, including the course description, based on the previously output text. The data generated off of each syllabus was saved into a pandas df, and then output to a CSV so I could easily analyze the output and update the prompts/temperature/other parameters to try to improve the AI output. The original filename was also saved as a reference point back to the source.

_Generate description snippet (note: generation\_bot.chatbot() is the chatbot defined to return class information including the generated class description):_
```python
#Function to generate a course description based on relevant content that was already extracted.
async def generate_description(text):
   ai_gen_bot = generation_bot.chatbot()
   ai_gen_bot.temperature = 1

   prompt = "Generate a course description using this content from the syllabus. Do not include class-specific information such as due dates, the instructor, etc. Also, do not include non-course information such as the university's mission statement and disability resources.\n"
   prompt += "\n### Syllabus Content ###\n"
   prompt += text
   prompt += "\n### End Syllabus Content ###\n"

   (response, input_tokens, output_tokens) = await ai_gen_bot.answer_question(prompt)

   input_cost = input_tokens / 1000 * INPUT_COST_PER_1K_TOKENS
   output_cost = output_tokens / 1000 * OUTPUT_COST_PER_1K_TOKENS
   total_cost = input_cost + output_cost
   print(response)
   return (response, total_cost)
```
_Parsing the description snippet:_
```python
# parse description
       try:
           # defaults in case values don't exist in output
           code = ''
           title = ''
           credits = ''
           desc = description
           outcomes = ''

           if('Course Code: ' in description):
               code = description.split('Course Code: ')[1].split('\n')[0]
           if('Course Title: ' in description):
               title = description.split('Course Title: ')[1].split('\n')[0]
           if('Credit Hours: ' in description):
               credits = description.split('Credit Hours: ')[1].split('\n')[0]
           if('Course Description: ' in description and 'Student Learning Outcomes:' in description):
               desc = description.split('Course Description: ')[1].split('Student Learning Outcomes:')[0]
           if('Student Learning Outcomes:' in description):
               outcomes = description.split('Student Learning Outcomes:')[1]
        
           # save to overall pdf df
           df.loc[len(df)] = [filename, extracted_text, output_tokens, title, desc, outcomes, code, credits, total_cost, pdf_time]
       except:
           print("Could not parse description")
           df.loc[len(df)] = [filename, extracted_text, output_tokens, '', description, '', '', 'X', total_cost, pdf_time]
```

#### 4. ‘Harmonize’ the descriptions so that each course has only one course description (and the other course information) using another OpenAI API call

This step ended up being a bit more complicated due to the nature of how the syllabi data was originally provided. It was relatively easy to get the descriptions for each individual syllabus, but the idea was to have multiple syllabi for one course and look at all of them to end up with one description. I managed to find a workable solution by using the AI output from step 3, so it did not require any additional API calls (which cost a small amount of money each time the calls are made). My solution was to create subdirectories based on the course code data point, and split up the data frame so each row became its own CSV file within the class code labeled-subdirectory. Most of the files did end up under the same subdirectory with this approach, and doing a manual clean up was easy and only took a few moments. This approach was still not ideal, but without a structured way of having the data provided, I needed to ensure that the ‘harmonization’ step would work and use the correct syllabi.

With the syllabi all sorted into neat subdirectories based on the class, I was able to use another chatbot class to read in the texts and output new course information data points using however many syllabi we had for each course (it varied from only one to up to about seven).

_Code snippet for sorting the data solution:_
```python
course_code = df.iloc[-1, 6]
       if course_code != '':
           path = extracted_path + course_code + '/'
           # Check if the directory exists, and create it if it doesn't
           if not os.path.exists(path):
               os.makedirs(path)  # This creates the directory and any required intermediate directories
           chunk_df.to_csv(path + filename + '_chunks.csv', index=False, escapechar='\\')
       else:
           #Don't put into a subdirectory associated with a course code if we weren't able to extract a course code; for manual review
           chunk_df.to_csv(extracted_path + filename + '_chunks.csv', index=False, escapechar='\\')
```

_Harmonization snippet (note: generation\_bot.chatbot() is the chatbot defined to return class information including the generated class description):_
```python
#Function to harmonize descriptions

def harmonize(descriptions):
   ai_gen_bot = harmonizer_bot.chatbot()
   ai_gen_bot.temperature = 1

   prompt = "Harmonize the following syllabi based on the information provided in each syllabus.\n"
   prompt += "\n### First Syllabus Content ###\n"
   prompt += descriptions[0]
   prompt += "\n### End Syllabus Content ###\n"

   #loop for the number of descriptions being sent for harmonization
   i = 1
   while i < len(descriptions):
       prompt += "\n### Next Syllabus Content ###\n"
       prompt += descriptions[i]
       prompt += "\n### End Syllabus Content ###\n"
       i +=1

   (response, input_tokens, output_tokens) = ai_gen_bot.answer_question(prompt)

   input_cost = input_tokens / 1000 * INPUT_COST_PER_1K_TOKENS
   output_cost = output_tokens / 1000 * OUTPUT_COST_PER_1K_TOKENS
   total_cost = input_cost + output_cost
   return (response, total_cost)
```

#### 5. Use additional API calls to re-style the original description

Sometimes the final description didn’t meet the content team’s requirements for what the description should sound like. After a lot of testing and analyzing output, I determined that trying to put all the styling and formatting requirements in the harmonizing API call resulted in worse descriptions. Therefore, I wrote up another chatbot class that defined specific rules that took in only the small course description text, which resulted in better written descriptions.

_Styling snippet (note: styling\_bot2.chatbot() is the chatbot defined to return class information including the generated class description):_
```python
#Function to stylize the description based on content team's feedback
def style(description):
   ai_gen_bot = styling_bot2.chatbot()
   ai_gen_bot.temperature = 1

   prompt = "Update the styling of this course description based on the rules provided.\n"
   prompt += "\n### Description Content ###\n"
   prompt += description
   prompt += "\n### End Description Content ###\n"

   (response, input_tokens, output_tokens) = ai_gen_bot.answer_question(prompt)

   input_cost = input_tokens / 1000 * INPUT_COST_PER_1K_TOKENS
   output_cost = output_tokens / 1000 * OUTPUT_COST_PER_1K_TOKENS
   total_cost = input_cost + output_cost
   return (response, total_cost)
```

#### 6. Check formatting and do some hard-coded formatting updates using nltk and regex; re-generate the description if necessary

There were some ‘requests’ in the rules I gave to the LLM that were not being reflected in the output, no matter how I phrased the requests in the prompt. A couple examples of these issues were not using Oxford commas (Oxford commas are not ASU brand approved), using common contractions, and trying to avoid sentences that were considered a call to action. These sentences started with verbs (like “Join us in this educational endeavor…"), so I added in some checks and simple regex to clean up the descriptions using both the Python packages nltk and re.

_Punctuation cleaning snippet:_
```python
#Function to update description with specific punctuation requests
def punctuation_updates(description):
   substitutions = [
       (', and', ' and'), #remove Oxford commas
       ('"', ''), #remove quotes around course names
       ("you will", "you'll"), #use contractions
       ("You will", "You'll"), #use contractions
       ('healthcare', 'health care'), #health care should be two words
       ('Healthcare', 'Health care'), #health care should be two words
       ('real-life', 'practical'), #don't use term real-life
       ('real-world', 'hands-on'), #don't use term real-world
       (r' \([A-Z]+\)', '') #don't include abbreviations such as (ABA)
   ]

   for pattern, replacement in substitutions:
       description = re.sub(pattern, replacement, description)
   return description
```

_Re-generating description based on some hard coded checks snippet:_
```python
#Function to check if a sentence starts with a verb or if it's too long
def check_sentences(text):
   #set the length of the total text description
   length_check = len(text.split())
   if length_check >= 110:
       #regenerate description and ask for shorter length
       print(f"Description exceeds 110 words in this text: '{text}'")
       return True

   #Tokenize the text into sentences
   sentences = sent_tokenize(text)
   for sentence in sentences:
       #Tokenize the sentence into words
       words = word_tokenize(sentence)
       if words:
           #Tag the words with part of speech
           pos_tags = nltk.pos_tag(words)
           #Check if the first word is a verb
           first_word, first_tag = pos_tags[0]
           if first_tag == 'VB':
               print(first_word, first_tag)
               print(f"Verb found at the beginning of the sentence: '{sentence}'")
               return True
           #manual check for most common verbs being generated in case first check doesn't grab it
           if first_word in ('Gain', 'Delve', 'Embark', 'Prepare', 'Enhance'):
               print(first_word, first_tag)
               print(f"Verb found at the beginning of the sentence: '{sentence}'")
               return True
   return False
 

def check_and_generate(df):
   #Add columns for regenerated descriptions if they don't exist
   if 'Regenerated Course Description 1' not in df.columns: #First Regenerated attempt
       df['Regenerated Course Description 1'] = None
   if 'Regenerated Course Description 2' not in df.columns: #Second Regenerated attempt
       df['Regenerated Course Description 2'] = None
   if 'Final Course Description' not in df.columns:
       df['Final Course Description'] = df['New Course Description 2']
   if 'Flagged' not in df.columns: #Flagged means it failed the verb check for a 3rd time and needs to be manually reviewed to remove CTA type phrases
       df['Flagged'] = 'False'
   #print(test)
   total_count = 0 #count to see how many need to be regenerated
   index_list = [] #list to keep track of the index of which rows needed to be regenerated

   #loop through DF and check if any sentences in the course description start with a verb
   for index, row in df.iterrows():
       text = row['New Course Description 2']
       check = check_sentences(text)
       if check: #If there is a verb at the start of a sentence in the description or length is too long
           total_count += 1
           regen_testing_list = [] #list for being able to add the regenerated descriptions to the dataframe

           regen_count = 0
           while regen_count < 3 and check: #We do not want to regenerate descriptions more than two times.
               new_description = regenerate_description(text)
               remove_commas = punctuation_updates(new_description[0])
               regen_count += 1
               regen_testing_list.append(remove_commas)
               check = check_sentences(remove_commas)

           if regen_testing_list == 2: #If we already regenerated the description two times, then flag it for manual review.
               flag = check_sentences(regen_testing_list[1])
               if flag:
                   df.loc[index, 'Flagged'] = 'True'
           index_list.append(index)

           #Assigning regenerated descriptions if they exist, as well as the final version
           if len(regen_testing_list)  == 1:
               df.loc[index, 'Regenerated Course Description 1'] = regen_testing_list[0]
               df.loc[index, 'Final Course Description'] = regen_testing_list[0]
           if len(regen_testing_list) == 2:
               df.loc[index, 'Regenerated Course Description 1'] = regen_testing_list[0]
               df.loc[index, 'Regenerated Course Description 2'] = regen_testing_list[1]
               df.loc[index, 'Final Course Description'] = regen_testing_list[1]
   return df
```

#### 7. Output final course information as a JSON object & CSV

Finally, the final output was saved. The JSON object could be used to feed into a future pipeline more easily, and the JSON was for an easier review process.

## Create Retrieval-Augmented Generation (RAG) for EdPlus Guru Data Project

_Main Python toolkits used: pandas, openai, tiktoken, re, BeautifulSoup_

**High level project context:** EdPlus employees at the call center often get asked questions by students, that they have to dig through many pages of data on a Guru repository to find answers to. The university requested that we use AI to enable the employees to find answers to these questions when they come up, where the answers are based on actual university data. Additionally, they requested the eventual application should also suggest related questions in case those were helpful to the student.

My role was mainly to upskill and learn about RAG. I wrote Python scripts that used ASU Guru data to set up a RAG system that could help call center employees get answers to questions students have, that currently lives in Guru.

**Project status:** Mini POC (proof of concept) completed; Example output was available in a Python notebook. This project did not have a real front end at the time I finished working on it.

### In depth project overview & process steps
 
#### 1. Read in JSON data from Guru cards & clean data

My coworker provided me with a compressed folder with JSON files he’d created from the data on Guru. The content he extracted from each page was still in html format, so the first step was to remove the html tags and do some general clean up on the text. I used python’s BeautifulSoup library to accomplish this. I tried to preserve some formatting such as trying to make it clear when any data was in a table format. 

_Code snippets:_
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

#Function to clean text
def replace_emojis(text):
   emoji_pattern = re.compile("["
                          u"\U0001F600-\U0001F64F" 
                          u"\U0001F300-\U0001F5FF"  
                          u"\U0001F680-\U0001F6FF"  
                          u"\U0001F1E0-\U0001F1FF"  
                          u"\U00002702-\U000027B0"
                          u"\U000024C2-\U0001F251"
                          u"\U0001F7E0-\U0001F7E3" 
                          "]+", flags=re.UNICODE)

   removed_emoji = emoji_pattern.sub('',text) #first remove emojis based on patterns
   cleaned_text = re.sub(r'\n\n+', '\n', removed_emoji) #remove double newlines

   #Chose to remove Click to Expand text because it occurs in 147 texts, and provides no value.
   remove_click_to_expand = re.sub(r'\nClick to Expand\n', '\n', cleaned_text) #remove any occurances of Click to Expand
   remove_pipe_w_nl = re.sub(r'(\n\|\n)', '\n', remove_click_to_expand)

   return remove_pipe_w_nl
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
           print(f"Error occurred at index {index} with content: {item['all_content'][:100]}")  #Limiting content display to first 100 characters
           print("Error details:", e)
           error_list.append(index)
```

#### 3. Set up RAG for the cleaned up texts & their vector embeddings

This was the most crucial step for this project, and was a technique I had not learned about in my HLT program (but it’s extremely useful and quite relevant in the industry today). RAG allows for an LLM that was not trained on specific data to add that data to the LLM, so that it can answer questions about something catered to a specific company (or whatever data you provide it). Learning about RAG was the biggest reason I was instructed to take this project on. The code works by generating vector embeddings for any query, and then grabbing the text chunks that are closest to it based on the cosine similarity between that query’s embedding and the embeddings it generated for each text (in this case the Guru text data). In this way the AI can provide tailored responses based on information it would otherwise not ‘know’, since it wasn’t trained on that data.

_Setting up RAG snippet:_
```python
class chatbot:
   def __init__(self, personality, cite_sources = False, keep_history = False, source_distance = 1, temperature=1):
       self.personality = personality
       self.cite_sources = cite_sources
       self.keep_history = keep_history
       self.source_distance = source_distance
       self.temperature = temperature
       personality = "Call Center"

       instruction = "Pretend you are a human being who works at the ASU Online Call Center. "
       instruction += "Your goal is to answer my questions about Arizona State University. "
       instruction += "Please be friendly, helpful, and informative. "
       instruction += "Do not say anything about being an AI or a language model in your responses. "
       instruction += "Do not say anything about getting information from the available sources, as that sounds like a robot and you are a human."
       instruction += "\nToday is " + date.today().strftime("%B %d, %Y") + "."
       instruction += " The current semester is Spring 2023."

       response = "Please enter your question."
       self.messages = [
           {"role": "user", "content": instruction},
           {"role": "assistant", "content": response},
       ]

   @property
   def cite_sources(self):
       return self._cite_sources

   @cite_sources.setter
   def cite_sources(self, cite_sources):
       if(cite_sources is True):
           print("Warning! Sources within the response may be incorrect.")
       self._cite_sources = cite_sources

   def create_context(
       self, question, df, max_len=1800, size="ada"
   ):
       """
       Create a context for a question by finding the most similar context from the dataframe
       """

       sources = []
       #Get the embeddings for the question
       q_embeddings = openai.Embedding.create(input=question, engine='text-embedding-ada-002')['data'][0]['embedding']
       #Get the distances from the embeddings
       df['distances'] = distances_from_embeddings(q_embeddings, df['embedding'].values, distance_metric='cosine')
       returns = []
       cur_len = 0

       #Sort by distance and add the text to the context until the context is too long

       for i, row in df.sort_values('distances', ascending=True).iterrows():
           #Add the length of the text to the current length
           cur_len += row['n_tokens'] + 4
           #If the context is too long, break
           if cur_len > max_len:
               break
           # If the distance is too far, break
           if row['distances'] > self.source_distance:
               break
           #Else add it to the text that is being returned
           text = row['content']

           if self.cite_sources:
               text += "\nSource: " + row['url']
           returns.append(text)
           source = (row['url'], row['distances'])
           sources.append(source)

       #Return the context
       return ("\n\n###\n\n".join(returns), sources)

   def answer_question(
           self,
           question,
           model="gpt-3.5-turbo",
           max_len=1800,
           size="ada",
           debug=False,
           max_tokens=150,
           stop_sequence=None
   ):
       #Answer question based on most similar context from dataframe texts
       (context, sources) = self.create_context(
           question,
           df,
           max_len=max_len,
           size=size,
       )

       #sources is a list of tuples, where the first element is the url and the second is the distance
       #we want to remove url duplicates and keep the closest distance
       #we also want to sort by distance
       sourceUrls = []
       sourceDistances = []
       for source in sources:
           if source[0] not in sourceUrls:
               sourceUrls.append(source[0])
               sourceDistances.append(source[1])
       sourcesSet = list(zip(sourceUrls, sourceDistances))

       #If debug, print the raw model response
       if debug:
           print("Context:\n" + context)
           print("\n\n")
       try:
           prompt = "Answer the question based on the provided content."
           prompt += "If the context is not relevant to the question, do not reference it."
           prompt += f" If the question can't be answered based on the context, start your answer with \"I'm not entirely sure. \" Please answer in complete sentences. \n\n Context: {context}\n\n---\n\nQuestion: {question}\nAnswer:"

           if self.cite_sources:
               prompt += "\nIn your answer, please state the source (url) provided in the section of the context you used to answer the question. Sections are separated by \"###\" and the source is listed at the end of the section. If you did not use a source, do not include a source."
           prompt += ". Your response should not be longer than a few sentences."

           tempMessages = self.messages.copy()
           tempMessages.append({"role": "user", "content": prompt})
           response = openai.ChatCompletion.create(
               model=model,
               messages=tempMessages,
               temperature=self.temperature
           )
           if self.keep_history:
               self.messages.append({"role": "user", "content": question})
               self.messages.append({"role": "assistant", "content": response["choices"][0]["message"]["content"]})
           return (response["choices"][0]["message"]["content"], sourcesSet)

       except Exception as e:
           print(e)
           return ("", "")

   def generate_questions(self, model="gpt-3.5-turbo", max_tokens=150):
       try:
           #Retrieve the most recent user question from the conversation history
           question = next((message['content'] for message in reversed(self.messages) if message['role'] == 'user'), None)
           if not question:
               raise ValueError("No user question found in the conversation history.")
           #Construct the prompt to generate related questions
           prompt = f"Generate three related questions based on the following user query: '{question}'"

           #Make the API call to generate related questions
           response = openai.ChatCompletion.create(
               model=model,
               messages=[{"role": "system", "content": "Provide three related questions to this question: " + question},  # System message to set the behavior of the assistant
                         {"role": "user", "content": question}],  #User's original question
               max_tokens=max_tokens,
               temperature=0.7
           )

           if self.keep_history:
               self.messages.append({"role": "assistant", "content": response["choices"][0]["message"]["content"]})
           return response["choices"][0]["message"]["content"], self.messages

       except Exception as e:
           print(e)
           return "", []
```

#### 4. Send some questions to the AI (any question) for testing

The next step was to write up some example questions to show the output. In addition to answering the question, I also programmed the small app to send back three related questions for every question the user asked (code snippet is above, in the function ‘generate_questions()’). The output was pretty accurate. This project did not end up getting used in any real ASU application as of this point, but it was an excellent learning experience where I gained insights into data cleaning and got to use openai’s vector embedding generation (and of course, RAG in general).

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
>Output: With an engineering degree, you can pursue various career paths depending on your specialization. Some potential careers include application developer, computer programmer, software architect, software developer, computer system analyst, director of engineering, engineering administrator, engineering project manager, systems and software manager, strategic technology developer, and more. These careers offer opportunities in industries such as software engineering, technology management, and sustainable engineering, among others.
>Related questions generated: 1. What are some specific job titles that someone with an engineering degree can pursue?\n2. Are there any industries or sectors that commonly hire individuals with an engineering degree?\n3. Can you provide examples of how an engineering degree can lead to a variety of career paths?

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
>Output (Pass/Fail status and answer to the questions):
>Pass - No, the Sociology BS program does not have accreditation. 
>Fail - The accrediting agency for the Leadership & Management, MLM program is AACSB (Association to Advance Collegiate Schools of Business). 
>Pass - Experiential learning opportunities for the Complex Systems Science, MS program include the option for students to engage in research or related activities, typically in the form of a capstone project, under the guidance of a faculty mentor. These opportunities allow students to apply their knowledge and skills in practical settings and gain hands-on experience in complex systems science. 
>Pass - An ideal candidate for the Social Entrepreneurship & Community Development Graduate Certificate is an individual who is currently working in or interested in working in any sort of community-facing role, seeking to be innovative in the creation or further development of programs. 
>Pass - Yes, the Biomimicry Graduate Certificate program is eligible for course recommendations and quick enrolls. 
 
## Automated Tool Assessment Form Completion Project
_Main Python toolkits used: pandas, openai, tiktoken, re, BeautifulSoup, HTMLParser_

**High level project context:** EdPlus employees have a tool assessment form that they fill out when validating whether a new tool should be adopted to the university or not. This takes up time while they go through the software’s website and answer many questions. The ask was for me to develop a very small POC on whether we could get this automated using AI.

My role was to write Python script(s) to answer questions on the provided tool assessment template, by setting up RAG and using information scraped from theoretically any tool’s website.

**Project status:** Small POC complete but could use some improvements (my internship came to an end before I could spend more time improving outputs and doing further testing). Output was available as a JSON and CSV for the team's review.

###In depth project overview & process steps

#### 1. Scrape website of a tool, given the name of the site and the name of the tool 

This was the only step that differed quite a bit from the second project (the RAG set up for ASU Guru data). It was also a challenge and something I raised a red flag about, in that trying to set up a web scraper to scrape any website isn’t very practical. Typically web scrapers are tailored for a specific site’s information you are trying to get. Nevertheless, I persisted in modifying a script we already had in our recipe book to scrape ASU data, and did testing on a handful of different sites to see what would work. When the scraper was successful, ultimately the results were promising. However, I ran into some sites where when I ran my script, nothing got scraped. Many sites are set up not to allow scraping, so this was not a surprise.

The code is structured in that it uses BeautifulSoup, Comment, deque, HTMLParser, and urlparse to scrape data, and clean it up at least somewhat. It saves a .txt file for each page it scrapes, under a subdirectory labeled with the name of the home page.

#### 2. Use OpenAI API to clean up the scraped texts

My experience from working on the Course Description project came in handy here. I found that using OpenAI to clean up the text in a similar fashion to how the relevant content extraction section of that project applied well to cleaning up the messy text. Normally I would have spent more time trying to have a clean output through BeautifulSoup or other similar Python library, but because the text was theoretically supposed to be scraped from any site, this worked as a generic approach to cleaning up text without seeing what that text looks like beforehand.

_Code snippet of the chatbot class to clean up text:_
```python
class chatbot:
   def __init__(self, keep_history = False, temperature=1):
       self.keep_history = keep_history
       self.temperature = temperature

       instruction = "You provide the most relevant topics from provided text, because the provided text was scraped from websites and is messy. "
       instruction += "The kind of content you want to include is information in complete sentences that conveys a clear message. Avoid including text "
       instruction += "in your response that is likely to have been scraped from a website header and footer, such as addresses, phone numbers, and common website "
       instruction += "footer elements like 'Join the team', 'Book a demo', 'Contact sales', etc. Try to look out for words that might have been buttons on a website "
       instruction += "that don't make sense to include as part of the main ideas."

       response = "I provide the most relevent information from text that I am provided. I focus on the main ideas of the text and avoid information "
       response += "that was likely scraped from website headers, footers, navigation menus, and website buttons."

       prompt = "Extract the relevant information from this text:"
       prompt += "\n### Text content ###\n"
       prompt += "html About | Insync Solutions  \u008f Email AI Chat AI for Customer Support Chat AI for Ecommmerce Agent AI InSync Browser Co-Pilot (beta) Why Insync  \u008f Technology Reporting & Insights Integrations Resources  \u008f About Blog Case Studies Contact sales Book a demo Contact sales Book a demo About We provide a cutting edge support automation solution with conversational AI for online lending and retail eCommerce companies. Join the team  Join the team  Join the team  About us Insyncai was founded by Raj Ramaswamy and Ashish Parnami with the mission to make Conversational AI more accessible and easy to implement for enterprises and without requiring any time or effort commitment from their side. Our powerful Conversational AI solution helps our clients drive sales, better customer acquisitions and scale their support operations at a fraction of the cost. Join the team  Meet the team Raj Ramaswamy CEO & Co-Founder  \u0099 Ashish Parnami CTO & Co-Founder  \u0099 Scott Sapire VP Sales  \u0099 Girish Nair Lead Architect  \u0099 Manish Jain Manager Data Science  \u0099 Anisha Jayan Tech Lead  \u0099 Prasanna Kumar K Editorial Project Manager  \u0099 Shrisha S Bhat Sn. Technical Product Manager  \u0099 Investors & Advisors Watertower Ventures  \u0099 ¡\u0086 BAM Ventures  \u0099 ¡\u0086 Foothill Ventures  \u0099 ¡\u0086 Arka Venture Labs  \u0099 ¡\u0086 Tuoc Luong  \u0099 ¡\u0086 Vivek Sharma  \u0099 ¡\u0086 Vishal Makhijani  \u0099 ¡\u0086 Jesse Bridgewater  \u0099 ¡\u0086 Contact us USA office Riverpark Tower, 333 W San Carlos St, San Jose, CA 95110 +1 888-291-0379 info@insyncai.com India office Indiqube Orion 4th Floor, 24th Main, HSR Layout, Bangalore-560102 India Interested in joining the team? Learn more  \u0092 USA office Riverpark Tower, 333 W San Carlos St, San Jose, CA 95110 +1 888-291-0379 info@insyncai.com India office Indiqube Orion 4th Floor, 24th Main, HSR Layout, Bangalore-560102 India Solutions Email AI Chat AI for Customer Support Chat AI for Ecommerce Agent AI Why Insync Technology Reporting & Insights Integrations Product About Blog Case Studies Privacy Policy Copyright 2024 @ InSync All Rights Reserved  Security Overview"
       prompt += "\n### End text content ###\n"

       example_response = "We provide a cutting edge support automation solution with conversational AI for online lending and retail eCommerce companies. "
       example_response += "Insyncai was founded by Raj Ramaswamy and Ashish Parnami with the mission to make Conversational AI more accessible and easy to implement for enterprises and without requiring any time or effort commitment from their side. Our powerful Conversational AI solution helps our clients drive sales, better customer acquisitions and scale their support operations at a fraction of the cost."

       second_prompt = "Extract the relevant information from this text:"
       second_prompt += "\n### Text content ###\n"
       second_prompt += "html Co-Pilot Solutions   Email AI Chat AI for Customer Support Chat AI for Ecommmerce Agent AI InSync Browser Co-Pilot (beta) Why Insync   Technology Reporting & Insights Integrations Resources   About Blog Case Studies Contact sales Book a demo Contact sales Book a demo Your Trusty Browser Co-Pilot Your AI-powered browser sidekick to navigate through complex web pages, sort through products, get product information, summarize articles, pdfs and more. The Co-Pilot uses advanced AI to understand the contents and uses generative AI to answer your questions in seconds. Install for Chrome  Get information on products in seconds The InSync Co-Pilot goes through entire web pages including embedded PDFs, so you can get all the information you need before you buy. Install for Chrome  Summarize articles, reviews snd more InSyncs Co-Pilot saves you time by summarizing long content articles, research, and more, so you can get the gist in seconds. Install for Chrome  Designed for shoppers, students, programmers & more.. InSyncs Co-Pilot runs on all kinds of web pages from E-commerce to Education to Q&A websites like Stack Overflow and more. Install for Chrome  Research @ the speed of thought InSyncs Co-Pilot helps you speed up your research by going through research documents, PDFs, etc and helping you get to the key information faster. Install for Chrome  InSync cares deeply about user safety and security and takes great care when handling your data. InSync does not retain or store any PII and all data is anonymized. Please do not share any personal information, including passwords, credit card or banking details. This plugin uses Artificial Intelligence (AI) and there may be inaccuracies or unintended bias in any results provided. By continuing, you agree to Open AI's and InSync's Terms & Privacy Policy . For any questions or feedback on Co-Pilot please contact us at copilot@insyncai.com . Get Always On - White Glove Service Get powerful business insights generated by our AI and enhanced by our Data Science team delivered to you on a weekly basis from your Insync AI Expert.   Book a demo USA office Riverpark Tower, 333 W San Carlos St, San Jose, CA 95110 +1 888-291-0379 info@insyncai.com India office Indiqube Orion 4th Floor, 24th Main, HSR Layout, Bangalore-560102 India Solutions Email AI Chat AI for Customer Support Chat AI for Ecommerce Agent AI Why Insync Technology Reporting & Insights Integrations Product About Blog Case Studies Privacy Policy Copyright 2024 @ InSync All Rights Reserved  Security Overview"
       second_prompt += "\n### End text content ###\n"

       second_example_response = "Your AI-powered browser sidekick to navigate through complex web pages, sort through products, get product information, summarize articles, pdfs and more. The Co-Pilot uses advanced AI to understand the contents and uses generative AI to answer your questions in seconds. "
       second_example_response += "Get information on products in seconds The InSync Co-Pilot goes through entire web pages including embedded PDFs, so you can get all the information you need before you buy. Summarize articles, reviews snd more InSyncs Co-Pilot saves you time by summarizing long content articles, research, and more, so you can get the gist in seconds. "
       second_example_response += "Install for Chrome  Designed for shoppers, students, programmers & more.. InSyncs Co-Pilot runs on all kinds of web pages from E-commerce to Education to Q&A websites like Stack Overflow and more. Research @ the speed of thought InSyncs Co-Pilot helps you speed up your research by going through research documents, PDFs, etc and helping you get to the key information faster. InSync cares deeply about user safety and security and takes great care when handling your data. InSync does not retain or store any PII and all data is anonymized. Please do not share any personal information, including passwords, credit card or banking details. This plugin uses Artificial Intelligence (AI) and there may be inaccuracies or unintended bias in any results provided."

       self.messages = [
           {"role": "user", "content": instruction},
           {"role": "assistant", "content": response},
           {"role": "user", "content": prompt},
           {"role": "assistant", "content": example_response},
           {"role": "user", "content": second_prompt},
           {"role": "assistant", "content": second_example_response},
       ]

   def answer_question(self, question, model="gpt-3.5-turbo"#, max_tokens=150
                       ):
       try:
           tokenizer = tiktoken.get_encoding("cl100k_base")
           #Add the user's question to the messages
           self.messages.append({"role": "user", "content": question})
           input_tokens = sum(len(tokenizer.encode(message["content"])) for message in self.messages)

           response = openai.ChatCompletion.create(
               model=model,
               messages=self.messages,
               temperature=self.temperature,
               #max_tokens=max_tokens
           )

           #Extract the response content
           response_content = response.choices[0].message.content
           output_tokens = len(tokenizer.encode(response_content))

           if self.keep_history:
               #Save the assistant's response to the history
               self.messages.append({"role": "assistant", "content": response_content})
           return response_content, input_tokens, output_tokens

       except Exception as e:
           print(f"Error during chat completion: {e}")
           #Return empty response and zero tokens in case of an exception
           return "", 0, 0
```

#### 3. Generate vector embeddings (using OpenAI) for the cleaned up text

This was very similar to how I set up vector embeddings in the second project I described above. I used OpenAI to generate vector embeddings in preparation for setting up RAG.

#### 4. Set up RAG for the cleaned up texts & their vector embeddings

I used a very similar set up for this project as I did for the second one. For this project I looked into adding HyDE (Hypothetical Document Embedding) as part of the process. By adding in HyDE, it is essentially asking the LLM to generate a document and adding it to the user query, and then creating vector embeddings from that instead of just the question. In this way, texts that are more relevant could rank higher as top results. For my small POC, I did not end up including the HyDE portion as part of the run, because I did not see better results when I added it in. This could use more experimentation, but my theory is that because it was looking for such specific information about the tool in question, the hypothetical document didn’t add enough relevant context.

_Code snippet including the chatbot class for setting up RAG, including HyDE:_
```python
class chatbot:
   def __init__(self, cite_sources = False, keep_history = False, source_distance = .3, temperature=1):
       self.cite_sources = cite_sources
       self.keep_history = keep_history
       self.source_distance = source_distance
       self.temperature = temperature

       instruction = "Your goal is to assess information about web tools, and use the relevant information to answer specific questions about the tool. "
       response = "Please enter your question."

       self.messages = [
           {"role": "user", "content": instruction},
           {"role": "assistant", "content": response},
       ]

   @property
   def cite_sources(self):
       return self._cite_sources

   @cite_sources.setter
   def cite_sources(self, cite_sources):
       if(cite_sources is True):
           print("Warning! Sources within the response may be incorrect.")
       self._cite_sources = cite_sources

   """
   def generate_hypothetical_document(self, question, model="gpt-3.5-turbo-instruct", temperature=1):
       system_prompt = "Even if you don't know the answer respond to this question with output from a typical AI tool website: "
       system_prompt += question
       print(f"Generating document with prompt: {system_prompt}")

       try:
           response = openai.Completion.create(
               engine=model,
               prompt=system_prompt,
               temperature=temperature,
               max_tokens=250,
               top_p=1.0,
               frequency_penalty=0.0,
               presence_penalty=0.0
           )

           return response.choices[0].text.strip()
       except Exception as e:
           print(e)
           return ""
   """

   def create_context(self, question, df, max_len=1800, size="ada"):
       """
       #Generate hypothetical document
       hypothetical_document = self.generate_hypothetical_document(question)
       print()
       print(f'Question: {question}')
       print(f'This is the hypothetical document: {hypothetical_document}')
       print()

       #combine hypothetical document with question for embeddings
       combined_input = hypothetical_document + ". " + question
       """
       sources = []

       #Get the embeddings for the combined input
       q_embeddings = openai.Embedding.create(input=question, engine='text-embedding-ada-002')['data'][0]['embedding']
       #Get the distances from the embeddings
       df['distances'] = distances_from_embeddings(q_embeddings, df['Embeddings'].values, distance_metric='cosine')

       returns = []
       cur_len = 0

       #Sort by distance and add the text to the context until the context is too long
       for i, row in df.sort_values('distances', ascending=True).iterrows():
           #Add the length of the text to the current length
           cur_len += row['n_tokens'] + 4
           #If the context is too long, break
           if cur_len > max_len:
               break
           #If the distance is too far, break
           if row['distances'] > self.source_distance:
               break
           #Else add it to the text that is being returned
           text = row['AI Cleaned Text']
           if self.cite_sources:
               text += "\nSource: " + row['URL']
           returns.append(text)
           source = (row['URL'], row['distances'])
           sources.append(source)
 
       #Return the context
       return ("\n\n###\n\n".join(returns), sources)

   def answer_question(
           self,
           question,
           df,
           tool_name,
           model="gpt-3.5-turbo",
           max_len=1800,
           size="ada",
           debug=False,
           max_tokens=150,
           stop_sequence=None
   ):
       #Answer question based on most similar context from dataframe texts
       (context, sources) = self.create_context(
           question,
           df,
           max_len=max_len,
           size=size,
       )
       #sources is a list of tuples, where the first element is the url and the second is the distance
       #we want to remove url duplicates and keep the closest distance
       #we also want to sort by distance
       sourceUrls = []
       sourceDistances = []
       for source in sources:
           if source[0] not in sourceUrls:
               sourceUrls.append(source[0])
               sourceDistances.append(source[1])
       sourcesSet = list(zip(sourceUrls, sourceDistances))

       #If debug, print the raw model response
       if debug:
           print("Context:\n" + context)
           print("\n\n")
       try:
           print('This is the context and question:')
           print(question)
           print()
           print(context)
           print()

           prompt = "Do your best to answer the question. I am providing some additional context to help answer the question, but if you are already "
           prompt += "familiar with the tool, feel free to provide information you know. "
           prompt += f"The name of the tool we need information about is {tool_name}."
           prompt += "If you are unsure about what the answer is, state that you are not entirely sure, and give a hypothetical or probable answer. "
           prompt += f"Additional context: {context} \n\n Question you need to answer: {question}"

           if self.cite_sources:
               prompt += "\nIn your answer, please state the source (url) provided in the section of the context you used to answer the question. Sections are separated by \"###\" and the source is listed at the end of the section. If you did not use a source, do not include a source."

           response = openai.ChatCompletion.create(
               model=model,
               messages=[{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": prompt}],
               temperature=self.temperature
           )
           if self.keep_history:
               self.messages.append({"role": "user", "content": question})
               self.messages.append({"role": "assistant", "content": response["choices"][0]["message"]["content"]})

           return (response["choices"][0]["message"]["content"], sourcesSet)

       except Exception as e:
           print(e)
           return ("", "")
```

#### 5. Send specific questions from the template to the AI to answer about the tool

I grabbed questions from the template and experimented with rephrasing them to try to get better output results. Ultimately the responses were reasonable and accurate.

_Code snippet:_
```python
    name_of_tool = 'Harmonize'
    validation_qs = [f'What is the license pricing structure of {name_of_tool}?', f'Briefly describe the tool {name_of_tool} and its functionality.',
                    f'Describe the educational value the tool {name_of_tool} might provide to students, if a university were to leverage it.',
                    f'List competitors to the tool {name_of_tool}.',
                    f'What would potential justification be for a university adopting {name_of_tool}?',
                    f'If provided, write some quotes or testimonials from a user or client of {name_of_tool}.',
                    f'What company developed the tool {name_of_tool}?',
                    f'How long as the tool or tech solution {name_of_tool} been in business?',
                    f'In which stage of development is the product {name_of_tool} (BETA or fully developed?)']

   employee = chatbot(keep_history=False)
   testing_list = []

   #Answer questions
   for question in validation_qs:
       response, sources = employee.answer_question(question, df=df, debug=False, tool_name = name_of_tool)
       print("Question:", question)
       print("Response:", response)
       print("Sources:", sources)
       testing_list.append(dict({'Question': question, 'Response': response, 'Sources': sources}))

   #Use AI to get user story response
   education_response = testing_list[2]['Response']
   user_story_q = f'''Based on your response: "{education_response}", rewrite these user stories with the [BLANK]s filled in:
                    1) As an instructor, I want students to understand [BLANK] so that they can become [BLANK].
                    2) As a student, I want [BLANK] so that [BLANK].
                    3) As an instructor, I want [BLANK] so that [BLANK].'''

   response, sources = employee.answer_question(user_story_q, df=df, debug=False, tool_name = name_of_tool)
   testing_list.append(dict({'Question': user_story_q, 'Response': response, 'Sources': sources}))
```

#### 6. Save and analyze output (mainly Questions & Answers)

The output was saved in both a CSV and JSON object. After analyzing the output, I would readjust the prompting or other aspects of the project.

_Some sample output:_
>Question: What is the license pricing structure of AI Chatbot?
>Response: The license pricing structure for the AI Chatbot offered by Sendbird consists of multiple plans, including a free trial, Starter Plan at $99/month, Pro Plan at $499/month, and an Enterprise Plan with custom pricing. The plans vary in features and message credits, with the option to scale based on the number of AI chatbots and message credits needed per month. Additionally, there are overage fees for exceeding the allocated message credits.
>Question: Briefly describe the tool AI Chatbot and its functionality.
>Answer: The AI Chatbot tool provided by Sendbird allows users to build a custom GPT (Generative Pre-trained Transformer) on their website and mobile apps to automate engagement, marketing, sales, and support with conversational AI. Users can tailor customer interactions using first-party data to deliver unparalleled AI customer service. The AI chatbot platform enables the creation of highly customized chatbots with no code directly from the Sendbird dashboard. It can be utilized for various purposes like lead generation, shopping assistance, customer service, legal assistance, recruiting, and healthcare. The features of the tool include a chatbot widget builder, widget theming, chatbot profile customization, white labeling, message streaming, typing indicator, user feedback, and integration with various AI knowledge sources. This tool aims to enhance customer satisfaction, increase efficiency, and improve customer support experiences by handling incoming requests effectively and reducing resolution times.