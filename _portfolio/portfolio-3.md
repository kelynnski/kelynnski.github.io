---
title: "Automated Tool Assessment Project"
excerpt: |
  This was a project I worked on as part of my internship at EdPlus at ASU. The task in this project was to use AI to fill out a "Tool Assessment" form that currently needs to be filled out manually. I was able to accompish this using web scraping, setting up RAG, and using Python's openai library to generate the answers to the questions in the form. The main NLP tasks this project relates to is Question/Answering and information retrieval.

collection: portfolio
---
 
## Automated Tool Assessment Form Completion Project
_Main Python toolkits used: pandas, openai, tiktoken, re, BeautifulSoup, HTMLParser_

**High level project context:** EdPlus employees have a tool assessment form that they fill out when validating whether a new tool should be adopted to the university or not. This takes up time while they go through the software’s website and answer many questions. Thus, I was asked to develop a very small POC to see if we could get this automated using AI. My role was to write Python script(s) to answer questions on the provided tool assessment template, by setting up RAG and using information scraped from theoretically any tool’s website.

**Project status:** Small POC complete but could use some improvements (my internship came to an end before I could spend more time improving outputs and doing further testing). Output was available as a JSON and CSV for the team's review.

### In depth project overview & process steps

#### 1. Scrape website of a tool, given the name of the site and the name of the tool 

This was the only step that differed quite a bit from the second project (the RAG set up for ASU Guru data). It was also a challenge and something I raised a red flag about, in that trying to set up a web scraper to scrape any website isn’t very practical. Typically web scrapers are tailored for a specific site’s information you are trying to get. Nevertheless, I persisted in modifying a script we already had in our recipe book to scrape ASU data, and did testing on a handful of different sites to see what would work. When the scraper was successful, ultimately the results were promising. However, I ran into some sites where when I ran my script, nothing got scraped. Many sites are set up not to allow scraping, so this was not a surprise.

The code is structured in that it uses BeautifulSoup, Comment, deque, HTMLParser, and urlparse to scrape data, and clean it up at least somewhat. It saves a .txt file for each page it scrapes, under a subdirectory labeled with the name of the home page.

#### 2. Use OpenAI API to clean up the scraped texts

My experience from working on the Course Description project came in handy here. I found that using OpenAI to clean up the text in a similar fashion to how the relevant content extraction section of that project applied well to cleaning up the messy text. Normally I would have spent more time trying to have a clean output through BeautifulSoup or other similar Python library, but because the text was theoretically supposed to be scraped from any site, this worked as a generic approach to cleaning up text without seeing what that text looks like beforehand.

You can view the chatbot used for this step on my [Chatbot Samples GitHub Repository](https://github.com/kelynnski/chatbot-samples/blob/main/relevent_text_chatbot.py).

#### 3. Generate vector embeddings (using OpenAI) for the cleaned up text

This was very similar to how I set up vector embeddings in my [RAG for Guru Data Project](portfolio-2.md). I used OpenAI to generate vector embeddings in preparation for setting up RAG.

#### 4. Set up RAG for the cleaned up texts & their vector embeddings

I used a very similar set up for this project as I did for the [RAG for Guru Data Project](portfolio-2.md). But for this project, I looked into adding HyDE (Hypothetical Document Embedding) as part of the process. By adding in HyDE, it is essentially asking the LLM to generate a document answering the user's query before retrieving and ranking the sources, and add that hypothetical document to the user query. I then create vector embeddings from that addition to the query, instead of just the original query alone. In this way, texts that are more relevant could rank higher as top results. For my small POC, I did not end up including the HyDE portion as part of the run, because I did not see better results when I added it in. This could use more experimentation, but my theory is that because it was looking for such specific information about the tool in question, the hypothetical document didn’t add enough relevant context.

You can view the chatbot used for this step on my [Chatbot Samples GitHub Repository](https://github.com/kelynnski/chatbot-samples/blob/main/tool_assessment_chatbot.py).

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
>Question: What is the license pricing structure of AI Chatbot?<br>
>Response: The license pricing structure for the AI Chatbot offered by Sendbird consists of multiple plans, including a free trial, Starter Plan at $99/month, Pro Plan at $499/month, and an Enterprise Plan with custom pricing. The plans vary in features and message credits, with the option to scale based on the number of AI chatbots and message credits needed per month. Additionally, there are overage fees for exceeding the allocated message credits.<br>
>Question: Briefly describe the tool AI Chatbot and its functionality.<br>
>Answer: The AI Chatbot tool provided by Sendbird allows users to build a custom GPT (Generative Pre-trained Transformer) on their website and mobile apps to automate engagement, marketing, sales, and support with conversational AI. Users can tailor customer interactions using first-party data to deliver unparalleled AI customer service. The AI chatbot platform enables the creation of highly customized chatbots with no code directly from the Sendbird dashboard. It can be utilized for various purposes like lead generation, shopping assistance, customer service, legal assistance, recruiting, and healthcare. The features of the tool include a chatbot widget builder, widget theming, chatbot profile customization, white labeling, message streaming, typing indicator, user feedback, and integration with various AI knowledge sources. This tool aims to enhance customer satisfaction, increase efficiency, and improve customer support experiences by handling incoming requests effectively and reducing resolution times.