import os

from groq import Groq

client = Groq(
    api_key="..."
)

# read "feature_description_and_values.json" file
import json
import datetime
import time
import requests
current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

# Get the current time to use in the filename
current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

# Create a filename with the current time
filename = f"response_{current_time}.txt"



with open("feature_description_and_values.json", "r") as json_file:
    data = json.load(json_file)
# data[Variablename] = {Type,description,[(value,value_label)]}

# My final task is to develop a model that can predict heart attack based on the features in the dataset.
# Here, I will use grok to clean the features and their values to make them ready for training the model.

# for each variable_name, feed the variable_name, type, description, and values to the GROQ query
# ask it to explain the importance (or inimportance) of the feature
# ask it to write either ACCEPTED (if the feature should be kept for training) or REJECTED (if the feature should be dropped)
# ask it to write a function that will take a value of the feature, and return a better value (if the feature should be kept) for example, 
# if the values are categories like 1 (yes) 2 (no)  3 (unknown), the function should return 1 for "yes", 0 for the rest
# in this cas, the function would be lambda x: 1 if x == 1 else 0
# Save the response to a text file

# keep only 10 first keys to test the code
# data = {k: data[k] for k in list(data)[:10]}

with open(filename, "w") as file:
# Iterate over each variable in the dataset

    for variable_name, variable_info in data.items():
        feature_type = variable_info["Type"]
        feature_description = variable_info["Description"]
        feature_values = variable_info["Values"]
        # Truncate feature values if there are more than 10
        if len(feature_values) > 10:
            truncated_values = feature_values[:3] + [["...", "..."]] + feature_values[-3:]
        else:
            truncated_values = feature_values
        # Prepare prompt for GROQ to explain the feature importance and suggest its usage in the specified format
        prompt = f"""
        We are building a machine learning model to predict coronary heart disease. The feature we're examining is:

        Variable Name: {variable_name}
        Type: {feature_type}
        Description: {feature_description}
        Values: {truncated_values}

        Values are in the format: [[value, value_label]]. For example, [["1", "yes"], ["2", "no"], ["3", "unknown"]].
        Only the values are training data. The labels are for human understanding only.


        Please provide the response in the following dictionary format:
        {{
            "explanation": "Explain the importance or inimportance of this feature in predicting coronary heart disease.",
            "accepted": True or False (should the feature be ACCEPTED or REJECTED for training) Be harsh and reject the feature if it's not clearly useful. Subtle correlations are not enough to keep the feature,
            "mapping_function": "lambda value: [a function that takes a value of the feature and returns a better version of the value if the feature is accepted (could also return the same value if unchanged), otherwise leave it as None]"
            "category": "Categorical" or "Numerical" (It is categorical if the order of the values does not matter (state number, and other arbitrary numbers are categorical), otherwise it is numerical)
        }}

        In the mapping_function, consider that the input is always a string, but transform it to the appropriate type if necessary. For example, if the feature is categorical, keep it as a string. If the feature is numerical, convert it to an integer or float.

        Example: If the values are categories like [["1", "yes"], ["2", "no"], ["3", "unknown"]], the function should return "1" for "yes" and "0" for the rest. In this case, the function would be:
        lambda x: 1 if x == 1 else 0

        Don't add any unecessary text. Just provide the response in the specified format.
        """
        
        # Query the model for the feature analysis
        while True:
            try:
                chat_completion = client.chat.completions.create(
                    messages=[
                    {
                        "role": "user",
                        "content": prompt,
                    }
                    ],
                    model="llama3-8b-8192",
                )
                break  # Exit the loop if the request is successful
            except requests.exceptions.HTTPError as e:
                if e.response.status_code == 429:
                    print("Received 429 Too Many Requests. Waiting for 10 seconds before retrying...")
                    time.sleep(10)
                else:
                    raise  # Re-raise the exception if it's not a 429 error
        
        file.write(f"Response for {variable_name}:\n")
        file.write(chat_completion.choices[0].message.content)
        file.write("\n" + "-"*80 + "\n")

        # Print the response for the current feature
        print(f"Response for {variable_name} saved in {filename}.\n")

