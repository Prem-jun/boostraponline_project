''' 
Decription: 
    - The script file is created referring to main_boost.py 
Arguments:
Output:

''' 
import json
import os
     
def main(folder_path,fileload):

    # Define the path to the JSON file
    filename = os.path.join(folder_path,fileload)

    # Open the JSON file
    with open(filename, 'r') as file:
        # Load the JSON data
        samples = json.load(file)
    chunk_size = [data['chunk_size'] for data in samples] 
    return 0
            
if __name__=='__main__':
    folder_path = './config_sim_data/wiebull/'
    filename = 'wiebullshape1n10000'
    filetype = '.json'
    main(folder_path,filename+filetype)