---
title: "Generative AI Course Description Project"
excerpt: |
  This was a project I worked on as part of my internship at EdPlus at ASU. The task in this project was to create a small POC, where the goal was to attempt to use course syllabi as input, and then use generative AI (openai in particular) to generate new course descriptions for ASU Online courses. This was the project I spent the majority of my time during my internship working on, as it was a complex project that involved many iterations and using openai to accomplish different types of NLP tasks. These tasks included using openai to extract information, summarize texts, and generate new texts based on specific prompts and rules.


collection: portfolio
---

_Main Python toolkits used: pandas, openai, re, nltk, pdfminer, tiktoken_

### Summary
**High level project context:** The goal was to be able to use existing course syllabi, and then use those syllabi to somehow generate new course descriptions based on the information in the syllabi. My role was to write Python scripts that use OpenAI’s APIs to generate course descriptions and other course information, based on the provided syllabi. At the start of this project, this was a personal challenge because of my lack of experience with LLMs or the Python openai library. However, it was a perfect opportunity for me to learn and grow through research and experimentation as I worked through the project.

**Project status:** The project was intended as a POC (proof of concept). By the end of the project the POC was completed, and the course descriptions are available on this site: https://courses.asuonline.asu.edu/

### In depth project overview & process steps
 
#### 1. Read in syllabi (pdfs only), scrape text from pdfs

I was given a link to a dropbox that contained a mess of syllabi provided by numerous ASU colleges. Most of the syllabi were PDF files, so the first challenge was to find a convenient way to extract the text from the PDFs (which after doing some quick research I found a solution in the Python library pdfminer).

_Extracting pdf text snippet:_
```python
#Extract text from PDFS. Re-written to accommodate weird issue with some PDFs
#not being extracted with extract_pages().
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

   prompt = "Extract the relevant information from the syllabus content. Do not include "
   pormpt += "class-specific information such as due dates, the instructor, etc. Also, "
   prompt += "do not include non-course information such as the university's mission statement and disability resources.\n"
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

   prompt = "Generate a course description using this content from the syllabus. Do not include "
   prompt += "class-specific information such as due dates, the instructor, etc. Also, do not include "
   prompt += "non-course information such as the university's mission statement and disability resources.\n"
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
           #Don't put into a subdirectory associated with a course code if we weren't able to extract
           #a course code; for manual review
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

_Hard coded checks snippet:_
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
```

I also needed to regenerate the description based on the rules coded above.

_Regenerate the description snippet:_
```python
def check_and_generate(df):
   #Add columns for regenerated descriptions if they don't exist
   if 'Regenerated Course Description 1' not in df.columns: #First Regenerated attempt
       df['Regenerated Course Description 1'] = None
   if 'Regenerated Course Description 2' not in df.columns: #Second Regenerated attempt
       df['Regenerated Course Description 2'] = None
   if 'Final Course Description' not in df.columns:
       df['Final Course Description'] = df['New Course Description 2']
   if 'Flagged' not in df.columns: #Flagged means it failed the verb check for a 3rd time
       #and needs to be manually reviewed to remove CTA type phrases
       df['Flagged'] = 'False'
   #print(test)
   total_count = 0 #count to see how many need to be regenerated
   index_list = [] #list to keep track of the index of which rows needed to be regenerated

   #loop through DF and check if any sentences in the course description start with a verb
   for index, row in df.iterrows():
       text = row['New Course Description 2']
       check = check_sentences(text)
       if check: #If there is a verb at the start of a sentence in the description or 
           #length is too long
           total_count += 1
           regen_testing_list = [] #list for being able to add the regenerated
           #descriptions to the dataframe

           regen_count = 0
           while regen_count < 3 and check: #We do not want to regenerate descriptions
           #more than two times.
               new_description = regenerate_description(text)
               remove_commas = punctuation_updates(new_description[0])
               regen_count += 1
               regen_testing_list.append(remove_commas)
               check = check_sentences(remove_commas)

           if regen_testing_list == 2: #If we already regenerated the description two
            #times, then flag it for manual review.
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

#### 8. Manual reviewing process

For each generation step, there was a lot of manual review and analysis involved. I spent a lot of time going over the output in each step, from the initial PDF scraping to ensure that the texts were properly scraped, to reviewing the information the AI considered 'relevant' in the information extraction step, and to every course description it gave as output. Throughout the process the different steps were saved in CSVs so I could conveniently compare the results, make notes on what improved (or worsened) with each change to prompts and parameter adjustments, and then try again. In this way I learned which little tweaks to the prompts  made huge differences (gaining experience in prompt engineering), when trying to cram too much into one prompt resulted in less optimal text generations, and even the value of creating a 'fake' conversation history with example responses that made it so the output was more uniform and resembled what we were looking for. Given the nature of the trying to generate course descriptions and what counted as a 'good' course description, a lot of it was quite subjective, and sometimes when I found the course descriptions to be satisfactory, the content team reviewing the output would not like how the AI phrased something (which guided much of the hard coded rules and checks I put in place). Because of this, the experimentation and multiple iterations of the manual review process were necessary.