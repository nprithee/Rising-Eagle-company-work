
#%%
import os
import base64
import pandas as pd
#Function to convert an image file to base64
def image_to_base64(image_path):
    with open(image_path, "rb") as image_file: 
        base64_data = base64.b64encode(image_file.read()).decode() 
        return base64_data



#List of image paths
image_paths= dataset["Image Path"]
# Initialize an empty DataFrame
df= pd.DataFrame(columns=["Filename"] + [f"Column(i+1)" for i in range(0, 32000, 32000)])

# Iterate through the image paths
for image_path in image_paths:
    if image_path.endswith("eg") or image_path.endswith(".png"): # Adjust file extensions as needed
        filename =os.path.basename(image_path)
        base64_data = image_to_base64(image_path)

        #Split the base64 data into chunks of 32,000 characters
        base64_chunks =[base64_data[i:i + 32000] for i in range(0, len(base64_data), 32000)]
        # Create a dictionary to represent the row
        row_dict={"Filename": filename} 
        for i, chunk in enumerate(base64_chunks):
            row_dict[f"Column(1+1)"]= chunk
        #Append the data to the DataFrame
        df=df.append(row_dict, ignore_index=True)








# %%
