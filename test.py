import pandas as pd
import numpy as np

my_lst = [1,2,3,4,5]
arr = np.array(my_lst)
print(type(arr))
print(arr.shape)
arr1 = pd.array([1, 2, None], dtype=pd.Int64Dtype())
print(arr1)

Data = '{"employee_name":"Vishesh","email":"visheshsinha801@gmail.com","job_profile":[{"title":"Team Lead","title2":"Sr. Dev"}]}'
df = pd.read_json(Data)
print(df)
print(df.to_json(orient="records"))
