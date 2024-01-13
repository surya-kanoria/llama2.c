from openai import OpenAI
import os
import json
import time
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)  # for exponential backoff
from tqdm import tqdm

data_dir = "./data/PreferenceDataset"
eval_dir = "./data/TrainDataset"
num_files = 18
client = OpenAI()
print(os.environ["OPENAI_API_KEY"])
instruction = """
"Evaluate two stories based on the following sets of criteria and indicate which story is better for each set. Provide only the story number as your response."
Criteria Sets:
		Set 1: Combine Grammar, Plot Coherence, Plot Logic, and More Sadness.
		Set 2: Combine Grammar, Plot Coherence, Plot Logic, and More Happiness.
		Set 3: Combine Grammar, Plot Coherence, Plot Logic, and Sophistication.
		Set 4: Combine Grammar, Plot Coherence, Plot Logic, and Humour
		Set 5: Combine Grammar, Plot Coherence, Plot Logic, and Twist-Rich Plot
		Set 6: Combine Grammar, Plot Coherence, Plot Logic, and Rich Fantasy
		Set 7: Combine Grammar, Plot Coherence, Plot Logic, and Positive Message
		Set 8: Combine Grammar, Plot Coherence, Plot Logic, and Dense Narrative
		Set 9: Combine Grammar, Plot Coherence, Plot Logic, and Higher character count
		Set 10: More Sadness
		Set 11: More Happiness
		Set 12: Sophistication
		Set 13: Humour
		Set 14: Twist-Rich Plot
		Set 15: Rich Fantasy
		Set 16: Positive Message
		Set 17: Dense Narrative
		Set 18: Higher character count.
		

Response Format:
Set 1: [Story 1/Story 2]
Set 2: [Story 1/Story 2]
Set 3: [Story 1/Story 2]
…
Set 18: [Story 1/Story 2]

Story Inputs:
Story 1: {}
Story 2: {}

Note: Provide your assessment directly following the format above. No additional explanation is required.

"""

@retry(wait=wait_random_exponential(min=20, max=40), stop=stop_after_attempt(6))
def completion_with_backoff(**kwargs):
    return client.chat.completions.create(**kwargs)

all_responses = []
for i in range(26, 39):
    print('Generating evaluations for file:', i)
    file_name = os.path.join(data_dir, f"dataset_{str(i)}.json")
    with open(file_name, "r") as f:
        # open the dataset for reading but keep it on disk with memmap
          stories_list = json.load(f)
          for stories in tqdm(stories_list):
              gpt_instruction = instruction.format(stories["story1"],stories["story2"])
              completion = completion_with_backoff(
                model="gpt-3.5-turbo",
                messages=[
                  {"role": "user", "content": gpt_instruction}
                ]
              )
              response = completion.choices[0].message.content
              # Expected response
              # response = 'Set 1: Story 2\nSet 2: Story 1\nSet 3: Story 2\nSet 4: Story 1\nSet 5: Story 2\nSet 6: Story 2\nSet 7: Story 1\nSet 8: Story 1\nSet 9: Story 1\nSet 10: Story 1\nSet 11: Story 2\nSet 12: Story 1\nSet 13: Story 2\nSet 14: Story 2\nSet 15: Story 2\nSet 16: Story 1\nSet 17: Story 2\nSet 18: Story 1'
              evaluation = {
                   "story1": stories["story1"],
                   "story2": stories["story2"],
                   "gpt_response": response
              }
              for r in range(1,19):
                  evaluation[f"set_{str(r)}"] = response.split(f"Set {str(r)}: Story ")[-1].split("\n")[0]
              all_responses.append(evaluation)
    output_file = os.path.join(eval_dir, f"dataset_{str(i)}.json")
    with open(output_file, "w") as f:
         json.dump(all_responses, f)