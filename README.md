# Locally_Sensetive_Hashing_-LSH-
A code for Locally Sensetive Hashing with option to use different similarity measures

The code takes the following commands from the command line:
      [-d str] -> Data file path
      [-s int] -> Random seed
      [-m str] -> Similarit measure (Jaccard similarity/ Cosine similarity/ Discrete Cosine Similarit) with commands (js/cs/dcs)
      
Example input:
python main.py -d /very/long/path/to/data.npy -s 2021 -m dcs
