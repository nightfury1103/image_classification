import os, sys
import pandas as pd
import numpy as np
import time
from concurrent.futures import ThreadPoolExecutor

from g4f.client import Client
from g4f.Provider.GeminiPro import GeminiPro


class APICollector:
    def __init__(self, name_datasets) -> None:    
        self.client = Client(
            api_key="AIzaSyDG_BXLX52SyTRw31yUHHwJolymVqkc7PQ",
            provider=GeminiPro
        )
        self.name_datasets = "datasets/INTEL/" + name_datasets + "/" + name_datasets
        self.prompt = "Given an X-ray image, describe the scene in the image. And what is the main object in the image among the following categories: (A) NORMAL, (B) PNEUMONIA. Answer it after character '(A)', '(B)'."
        
    def call_api(self, path):
        res = np.nan
        while res is np.nan:
            try:
                response = self.client.chat.completions.create(
                    model="gemini-pro-vision",
                    messages=[{"role": "user", "content": self.prompt}],
                    image=open("/home/huy/Desktop/HCMUS/image_classification/PNEUMONIA_0.jpeg", "rb")  
                )
                res = response.choices[0].message.content
            except Exception as e:
                res = np.nan

            print(res)
            # time.sleep(5)
        return response.choices[0].message.content
        
    def load_datasets(self):
        labels = []
        images = []
        for label in os.listdir(self.name_datasets):
            path = os.path.join(self.name_datasets, label)
            if os.path.isdir(path):
                for image in os.listdir(path):
                    image_path = os.path.join(path, image)
                    
                    labels.append(label)
                    images.append(image_path)
        df = pd.DataFrame({
            'label': labels,
            'image': images
        })
        df['description'] = np.nan          
        self.df = df.sample(frac=1).reset_index(drop=True)
    
    def load_remain(self):
        df = pd.read_csv('/home/huy/Desktop/HCMUS/image_classification/description_data/INTEL/test/describe_1.csv', index_col=0)
        self.df = df[df.description.isna()]
    
    def get_describe(self):
        # cal multiprocess to get description
        num_cores = 15
        with ThreadPoolExecutor(max_workers=num_cores) as executor:
            futures = []
            index_to_path_map = {}
            for i, path in enumerate(self.df['image']):
                # Store index for future reference
                future = executor.submit(self.call_api, path)
                futures.append(future)
                index_to_path_map[future] = self.df['image'].index[i]

            # Wait for all futures and update DataFrame accordingly
            for future in futures:
                res = future.result()
                index = index_to_path_map[future]
                # Update the DataFrame directly at the correct index
                self.df.at[index, 'description'] = res
                print(f"Updated index {index} with result: {res}")
            
    def run(self):
        self.load_datasets()
        self.get_describe()
        return self.df
    
if __name__ == "__main__":
    name_datasets = "seg_test"
    api = APICollector(name_datasets)
    df = api.run()
    df.to_csv(f"/home/huy/Desktop/HCMUS/image_classification/description_data/INTEL/test/describe_2.csv", index=True)
    
        
    

