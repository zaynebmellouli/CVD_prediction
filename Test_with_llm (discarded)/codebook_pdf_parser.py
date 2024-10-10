# the goal of this file is to parse a pdf and scrape the useful text from it

# import PyPDF2

# # load "codebook15_llcp.pdf"
# pdf_file = open("codebook15_llcp.pdf", "rb")
# pdf_reader = PyPDF2.PdfReader(pdf_file)

# # save the text from the pdf to a txt file
# pdf_text = ""
# for page_num in range(len(pdf_reader.pages)):
#     page = pdf_reader.pages[page_num]
#     pdf_text += page.extract_text()

# with open("codebook15_llcp.txt", "w") as txt_file:
#     txt_file.write(pdf_text)

# read the text from the txt file

with open("codebook15_llcp.txt", "r") as txt_file:
    pdf_text = txt_file.read()

dict_data = {}

# dict_data[Variablename] = {Type,description,[(value,value_label)]}

while True:
    # find the next variable
    # find a string as follows : "Type:  Num"
    type_start = pdf_text.find("Type:  ")
    if type_start == -1:
        break
    type_end = pdf_text.find("\n", type_start)
    if type_end == -1:
        break
    type = pdf_text[type_start + len("Type:  "):type_end]
    # strip it from " " and "\n"
    type = type.strip()

    pdf_text = pdf_text[type_end:]

    # find a string as follows : "Variable Name:  _STATE"
    var_name_start = pdf_text.find("Variable Name:  ")
    if var_name_start == -1:
        break
    var_name_end = pdf_text.find("\n", var_name_start)
    if var_name_end == -1:
        break
    var_name = pdf_text[var_name_start + len("Variable Name:  "):var_name_end]
    var_name = var_name.strip()
    pdf_text = pdf_text[var_name_end:]

    # find a string as follows
    # Description:  Is this a cellular telephone?  (Telephone service over the internet counts as landline service (includes Vonage, Magic 
    # Jack and other home- based phone services).)[Read only if necessary: “By cellular (or cell) telephone we mean a 
    # telephone that is mobile and usable outside of your neighborhood.”]  
    # Value

    desc_start = pdf_text.find("Description:  ")
    if desc_start == -1:
        break
    desc_end = pdf_text.find("Value", desc_start)
    if desc_end == -1:
        break
    desc = pdf_text[desc_start + len("Description:  "):desc_end]
    desc = desc.strip()
    pdf_text = pdf_text[desc_end:]

    if var_name not in dict_data:
        dict_data[var_name] = {"Type": type, "Description": desc, "Values": []}

    # find string as follows
    # Percentage  
    # 1100 Completed Interview  375,059 84.96 77.74 
    # 1200 Partial Complete Interview  66,397 15.04 22.26 

    percentage_start = pdf_text.find("Percentage  Weighted")
    if percentage_start == -1:
        break
    percentage_end = pdf_text.find("\n", percentage_start)
    if percentage_end == -1:
        break
    pdf_text = pdf_text[percentage_end:]
    percentage_start = pdf_text.find("Percentage")
    if percentage_start == -1:
        break
    percentage_end = pdf_text.find("\n", percentage_start)
    if percentage_end == -1:
        break
    pdf_text = pdf_text[percentage_end:]
    while True:
        # find a string as follows : "1100 Completed Interview  375,059 84.96 77.74"
        # take the first number (up to the first space) as the value
        # take the rest of the string as the value label (up to the double space)

        # first check that this line is not just a " \n"
        if pdf_text[:2].strip() == "" or pdf_text.startswith("CODEBOOK REPORT"):
            break

        value_end = pdf_text.find(" ")
        if pdf_text[value_end + 1:value_end + 3] == "- ":
            value_end = pdf_text.find(" ", value_end + 3)
        if value_end == -1:
            break
        value = pdf_text[:value_end]
        value = value.strip()
        # print(value)
        if "HIDDEN" in value:
            dict_data[var_name]["Values"].append((0, "HIDDEN"))
            break
        if "Notes" in value:
            # go to next line
            end_of_line = pdf_text.find("\n")
            if end_of_line == -1:
                break
            pdf_text = pdf_text[end_of_line + 1:]
            continue
        value_label_end = pdf_text.find(" ", value_end + 1)
        while value_label_end != -1 and not pdf_text[value_label_end + 1].isdigit():
            value_label_end = pdf_text.find(" ", value_label_end + 1)
        if value_label_end == -1:
            break
        value_label = pdf_text[value_end + 1:value_label_end]
        value_label = value_label.strip()
        # print(value_label)
        # break
        end_of_line = pdf_text.find("\n")
        if end_of_line == -1:
            break
        pdf_text = pdf_text[end_of_line + 1:]

        if value not in [v[0] for v in dict_data[var_name]["Values"]]:
            dict_data[var_name]["Values"].append((value, value_label))
    # break


# save the data to a json file
import json

with open("feature_description_and_values.json", "w") as json_file:
    json.dump(dict_data, json_file)

