# Generate Data Spring26 Member Challenge

Congratulations! If you are receiving this challenge, you have passed the initial screening for being a member of the Data Branch, and you are onto Round 2! Below is a Technical Challenge that you must complete to the best of your ability. Be as creative as you can and really think outside the box. Challenge yourself, even if you’re not sure about something!

## Overview: 
This task is meant to test your ability to function in a real-world data scenario. It will teach you how to:
- **Deal with messy data.** This data is intentionally inconsistent to simulate real ticket systems. This “messiness” may be in terms of formatting, labels, or anything else. 
- **Understand and apply embeddings.** You must convert natural language (i.e. English) into numerical vectors using an embedding model of your choice.
- **Identify duplicates to remove.** Real-life workflows often require removal of duplicates based on logs, tasks, etc. This is meant to mimic this scenario. 
- **Justify analytical decisions.** In addition to computing the similarity, you must also
  - Explain why 2 tasks are related
  - Form meaningful clusters
  - Interpret any outliers
- **[OPTIONAL] Create a frontend to display duplicate tasks.** For those who want to go above and beyond.

**REMEMBER**: You are NOT building a model. You are utilizing existing tools to analyze data.

If there is anything else you have questions about, do not hesitate to reach out to us!
- Vichu Selvaraju selvaraju.v@northeastern.edu
- Nandeenee Singh singh.nand@northeastern.edu

## Challenge Instructions:
- **Retrieve the task cards using the Trello API.** The dataset is not provided to you directly. You must pull, parse, and analyze cards from a live Trello board.
  - First add the board to your workspace: https://trello.com/invite/b/6918f853b1842c1047085df4/ATTI4a257d1937641cd9a1478ac634c54def4DE3D00C/aguenta-tasks
  - API KEY: e5244381f2503a3dddbb64ac6f57d947
  - Retrieve your token using this route (make sure to edit the name field with camel case): https://trello.com/1/authorize?expiration=never&scope=read&response_type=token&name=[YOUR-NAME]&key=e5244381f2503a3dddbb64ac6f57d947
  - Make sure to store your key and token securely
  - **You should only be pulling data from the trello board we provided for you**
- **Export the raw API response.** You can use a file of your choice (JSON, CSV, txt, Excel)
  - Remember, the format you choose will impact the parsing of the fields later on. 
- **Parse the card fields.** Each Trello card has metadata that you must extract to determine important information about each task.
  - You must clean the text and prepare it for embedding. It should be easily readable.
- **Generate Text Embeddings.** Use an embedding model to convert each card into a numerical vector. We would suggest using the Sentence Transform python library, but you are not required to do so!
- **Compute Similarity.** For each card, compute similarity with every other card. Remember, your goal is to identify: 
  - Duplicate tasks
  - Near-duplicate tasks
  - Logically similar clusters
- **[OPTIONAL] Create a frontend to display duplicate tasks.** There are no explicit instructions for this step, just do what you think will convey this data adequately. 

## Submission Instructions

1. Create a new branch for your work: `git checkout -b [branch-name]`
2. Complete the challenge in your branch
3. Commit and push your work regularly
4. When ready, create a Pull Request from `[branch-name]` to `main`
5. **DO NOT MERGE** - Leave the PR open for review
6. Add any additional documentation or explanation in the PR description
