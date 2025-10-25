import os
import streamlit as st
import json
import time
from openai import OpenAI
client = OpenAI()
from io import BytesIO
import pandas as pd

os.environ['OPENAI_API_KEY'] = st.secrets['OPENAI_API_KEY']


if __name__ == "__main__":
    d = pd.read_csv('political_social_media.csv', encoding_errors= "ignore")
    result = d.reset_index().to_json(orient="records")
    data = json.loads(result)
    # for key, value in data.items():
    #     print(key, value)

    tweets = [row.get("text", "") for row in data]
    system_message = {"role": "system", "content": "write a tweet"}
    data = [[system_message, {"role": "user", "content": t} ] for t in tweets]

    # print(data)

    my_file = BytesIO()
    for m in data:
        my_file.write((json.dumps({"messages": m}) + "\n").encode('utf-8'))

    my_file.seek(0)
    training_file = client.files.create(
        file=my_file,
        purpose="fine-tune",
    )

    # while True:
    #     try:
    #         job = openai.FineTuningJob.create(training_file=training_file.id, model="gpt-3.5-turbo")
    #     except Exception as e:
    #         print(e)
    #         print("Trying again in ten seconds....")
    #         time.sleep(10)

    start = time.time()

    while True:
        ftj = client.fine_tuning.jobs.retrieve("ftjob-fVHiKTzMwuqei6v2AyEtjygd")
        if ftj.fine_tuned_model is None:
            print(f"Waiting for fine-tuning to complete... Elapsed: {time.time() - start}", end="\r", flush=True)
            time.sleep(10)
        else:
            print("\n")
            print(ftj.fine_tuned_model, flush=True)
            break