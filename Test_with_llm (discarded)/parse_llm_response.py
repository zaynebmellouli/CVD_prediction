
with open("response_second_try.txt", "r") as file:
    response = file.read()

# file is in this format : 

# Response for _STATE:
# {
#     "explanation": "The feature _STATE provides information about the patient's location, which can potentially be related to coronary heart disease trends, healthcare infrastructure, and lifestyle factors. However, without additional context, this feature is not clearly useful in predicting coronary heart disease.",
#     "accepted": False,
#     "mapping_function": None,
#     "category": "Categorical"
# }
# --------------------------------------------------------------------------------
# Response for FMONTH:
# {
#     "explanation": "The FMONTH feature appears to be a categorical feature representing the month of the year. The prediction of coronary heart disease does not seem to be strongly related to the month of the year, as it does not directly impact the progression of the disease.",
#     "accepted": False,
#     "mapping_function": None,
#     "category": "Categorical"
# }
# --------------------------------------------------------------------------------

# result_dict[variable_name] = {"accepted": accepted, "mapping_function": mapping_function, "category": category}

result_dict = {}

while True :
    # find "Response for ":
    response_start = response.find("Response for ")
    if response_start == -1:
        break
    response_end = response.find(":", response_start)
    if response_end == -1:
        break
    variable_name = response[response_start + len("Response for "):response_end].strip()
    # print(variable_name)
    response = response[response_end:]
    # find "accepted": 
    accepted_start = response.find('"accepted": ')
    if accepted_start == -1:
        break
    accepted_end = response.find(",", accepted_start)
    if accepted_end == -1:
        break
    accepted = response[accepted_start + len('"accepted": '):accepted_end].strip()
    # print(accepted)
    response = response[accepted_end:]

    # if accepted == "False":
    #     result_dict[variable_name] = {"accepted": False, "mapping_function": None, "category": None}
    #     continue

    # find "mapping_function":
    mapping_function_start = response.find('"mapping_function": ')
    if mapping_function_start == -1:
        break

    # find "category": " as the end of the mapping function
    category_start = response.find('"category": "')
    if category_start == -1:
        break
    mapping_function = response[mapping_function_start + len('"mapping_function": '):category_start].strip()
    # remove going back to line from mapping function
    mapping_function = mapping_function.replace("\n", "")
    response = response[category_start:]
    category_end = response.find('"')
    if category_end == -1:
        break
    category = "Categorical" if response[len("category")+5] in ["C","c"] else "Numerical"
    # print(category)
    result_dict[variable_name] = {"accepted": accepted, "mapping_function": mapping_function, "category": category}


print(result_dict)

# save it as csv with tabs
import csv

with open("result_dict.csv", "w") as file:
    writer = csv.writer(file, delimiter='\t')
    writer.writerow(["Variable Name", "Accepted", "Mapping Function", "Category"])
    for variable_name, variable_info in result_dict.items():
        writer.writerow([variable_name, variable_info["accepted"], variable_info["mapping_function"], variable_info["category"]])
    