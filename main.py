# Importing packages

import argparse
import numpy as np
import time
from scipy.sparse import csc_matrix
import itertools
import os
from datetime import datetime

# Create the parser and adding the arguments to get input to the algorithm 

parser = argparse.ArgumentParser()

# Now to add the arguments
# Use args.s / args.d / args.m / to access the inputs
parser.add_argument('-d', type=str, default=False)
parser.add_argument('-s', type=int, default=0)
parser.add_argument('-m', choices=['js', 'cs', 'dcs'])

# Parse and print the results
args = parser.parse_args()

#print(args.s)
#print(args.d)
#print(args.m)

# Setting the seed
np.random.seed(args.s)

#################################################################################
# Starting the time to assess the time

start_time = time.time()
now = datetime.now()

current_time = now.strftime("%H:%M:%S")
print("Current Time =", current_time)
#################################################################################
# Loading the data
original_data  = np.load(args.d)

movie_number = len(np.unique(original_data[:,1]))
user_number = len(np.unique(original_data[:,0]))

#################################################################################

# Defining some parameters

permutation_number = 95
projection_number = 114
band_size = 19
band_size_projection = 6
js_threshold = 0.5
cs_threshold = 0.73
dcs_threshold = 0.73
#################################################################################

# Defining the functions 


def Permutation_function(per_num = 120): # Makes a matrix consisting of the index values for each permutation (here 17770 by 120)
    permutation_matrix = np.empty([movie_number, per_num])
    rng = np.random.default_rng(seed=args.s)
    for i in np.arange(per_num):
        permutation_matrix[:,i] = rng.permutation(np.arange(movie_number))
    return permutation_matrix

def Signature_matrix(permutation_matrix,original_matrix , per_num = 120):
    signature = np.empty([per_num, user_number])
    for permutation in np.arange(per_num):
        permute_matrix = original_matrix[permutation_matrix[:,permutation],:]
        signature[permutation,:] = (permute_matrix!=0).argmax(axis=0)
    return signature

def Band_maker(signature,b):
    r = int(len(signature) / b)
    subvecs = []
    for i in range(0,len(signature),r):
        subvecs.append(signature[i : i + r])
    return subvecs

#def Band_maker(signature,b):
#    return np.split(signature, b, axis = 0)

def Create_bands(signature,band_size):
    bands_list = []

    for i in range(signature.shape[1]):
        bands_list.append(Band_maker(signature[:,i], band_size))

    return bands_list

def LocallySensetiveHashing(bands_list, band_size, limit= 200):
    user_numbers = user_number
    
    
    
    candidates = set()
    
    for i in np.arange(band_size):
        buckets = {}
        for user in np.arange(user_numbers):
            hash_value = hash(tuple(bands_list[i][:,user]))
            if hash_value in buckets.keys():
                buckets[hash_value].append(user)
            else:
                buckets[hash_value] = [user]
        for bucket_users in buckets.values():
            if 1 < len(bucket_users) < limit:
                #for i in range( len(bucket_users)-1 ):
                #    for j in range(i, len(bucket_users)-1):
                #        sorted_val = np.sort(np.array([bucket_users[i], bucket_users[j+1]]))
                        #print(sorted_val)
                #        candidates.add( (sorted_val[0], sorted_val[1]) )
                
                
                for combination in itertools.combinations(bucket_users,2):
                    candidates.add(combination)
    return list(candidates)

def Jaccard_similarity(sparse_matrix, candidate_one, candidate_two):
    person_one = np.nonzero(sparse_matrix[:,candidate_one].toarray())[0] # get the indexes of candidates
    person_two = np.nonzero(sparse_matrix[:,candidate_two].toarray())[0]
    return len(np.intersect1d(person_one,person_two)) / len(np.union1d(person_one,person_two))

def Random_projections(sparse_matrix,number_of_projections):
    projection_matrix = np.random.randn(sparse_matrix.shape[0], number_of_projections)
    out_matrix = np.empty([sparse_matrix.shape[1],number_of_projections])
    for user in range(0,sparse_matrix.shape[1]):
        one_user = sparse_matrix[:,user].toarray().T
        out_matrix[user,:] = np.sign(np.dot(one_user,projection_matrix))

    return out_matrix.T

#def Random_projections(sparse_matrix,number_of_projections):
#    out_matrix = np.empty([number_of_projections,sparse_matrix.shape[1]])
#    projection_matrix = np.random.randn(sparse_matrix.shape[0],number_of_projections)
#    for user in range(0,sparse_matrix.shape[1]):
#        one_user = sparse_matrix[:,user].toarray().T
#        out_matrix[:,user] = (np.dot(one_user,projection_matrix) > 0).astype(int)
#    return out_matrix
def Cosine_similarity(sparse_matrix,candidate_one,candidate_two):
    a = sparse_matrix[:,candidate_one].toarray()
    b = sparse_matrix[:,candidate_two].toarray()
    ab = np.dot(a.T, b)[0][0]
    cos_ab = ab / (np.linalg.norm(a) * np.linalg.norm(b))
    alpha = np.degrees(np.arccos(cos_ab))
    cossim = 1 - alpha / 180
    return(cossim)
#################################################################################

# For Jaccard Similarity

if args.m == "js":
    print("Implementing the Jaccard Similarity")
    print(f"Using {permutation_number} permutations, {band_size} bands and {round(permutation_number/band_size)} rows per band.")
    


    

    #  /////// Make a sparse matrix  \\\\\\\
    sparse_matrix = csc_matrix((original_data[:,2],(original_data[:,1]-1,original_data[:,0]-1)),shape=(movie_number, user_number))

    print(f"Sparse Matrix created.  [1/5]")
    #  /////// Making the signatures  \\\\\\\
    permutation_indecies = Permutation_function(per_num = permutation_number)

    signatures = Signature_matrix(permutation_matrix= permutation_indecies, original_matrix = sparse_matrix, per_num = permutation_number)

    print(f"Signatures created.  [2/5]")
    #  /////// Making the bands  \\\\\\\\

    banded_signatures = Band_maker(signatures,band_size) 

    print(f"Bands are created.  [3/5]")
    #  /////// Finding the candidates \\\\\\\

    candidates = LocallySensetiveHashing(banded_signatures, band_size, limit = 30)

    print(f"Candidates are found. We found {len(candidates)} candidates [4/5]")
    # /////// Finding the matches and making the output file \\\\\\\\\

    count = 0
    filename = "js.txt"

    if os.path.exists(filename):
	    os.remove(filename)

    for candidate_pair in candidates:
        if time.time() - start_time > 30 *60:
                print("30 minutes passed. Terminating the algorithm early")
                break
        if Jaccard_similarity(sparse_matrix,candidate_pair[0], candidate_pair[1]) > js_threshold:
            sorted = np.sort(candidate_pair)
            out = str(str(sorted[0])+ "," + str(sorted[1]))
            f = open(filename, "a+")
            if count != 0:
                f.write("\n")
            f.write(out)
            f.close()
            count += 1
            

    

    print(f"We found {count} similar pairs using Jaccard similarity")
    print(f"Jaccord Similariy algoritm successfully ran. [5/5]")


#################################################################################
# Cosine Similarity

if args.m == "cs":
    print("Implementing the Cosine Similarity")
    print(f"Using {projection_number} permutations, {band_size_projection} bands and {round(projection_number/band_size_projection)} rows per band.")

    #  /////// Make a sparse matrix  \\\\\\\
    sparse_matrix = csc_matrix((original_data[:,2],(original_data[:,1]-1,original_data[:,0]-1)),shape=(movie_number, user_number))
    print(f"Sparse Matrix created.  [1/5]")

    #  /////// Make random projections  \\\\\\\

    random_projections = Random_projections(sparse_matrix = sparse_matrix, number_of_projections = projection_number)
    print(random_projections.shape)
    print(f"Random projections created.  [2/5]")
    #  /////// Make the bands  \\\\\\\
    banded_signatures = Band_maker(random_projections,band_size_projection) 

    print(f"Bands created.  [3/5]")
    #  /////// Finding the candidates \\\\\\\

    candidates = LocallySensetiveHashing(banded_signatures, band_size_projection)

    print(f"Candidates are found. We found {len(candidates)} candidates [4/5]")

    # /////// Finding the matches and making the output file \\\\\\\\\

    count = 0
    filename = "cs.txt"

    if os.path.exists(filename):
	    os.remove(filename)


    for candidate_pair in candidates:
        if time.time() - start_time > 30 *60:
                print("30 minutes passed. Terminating the algorithm early")
                break
        if Cosine_similarity(sparse_matrix,candidate_pair[0], candidate_pair[1]) > cs_threshold:
            sorted = np.sort(candidate_pair)
            out = str(str(sorted[0])+ "," + str(sorted[1]))
            f = open(filename, "a+")
            if count != 0:
                f.write("\n")
            f.write(out)
            f.close()
            count += 1
            
    

    print(f"We found {count} similar pairs using Cosine similarity")
    print(f"Cosine Similariy algoritm successfully ran. [5/5]")


#################################################################################
# Discrete Cosine Similarity
if args.m == "dcs":
    print("Implementing the Discrete Cosine Similarity")
    print(f"Using {projection_number} permutations, {band_size_projection} bands and {round(projection_number/band_size_projection)} rows per band.")
    #  /////// Make a sparse matrix  \\\\\\\

    original_data[2] = 1 # Make ratings all equal to 1
    sparse_matrix = csc_matrix((original_data[:,2],(original_data[:,1]-1,original_data[:,0]-1)),shape=(movie_number, user_number))
    print(f"Sparse Matrix created.  [1/5]")

    #  /////// Make random projections  \\\\\\\

    random_projections = Random_projections(sparse_matrix = sparse_matrix, number_of_projections = projection_number)

    print(f"Random projections created.  [2/5]")
    #  /////// Make the bands  \\\\\\\
    banded_signatures = Band_maker(random_projections,band_size_projection) 

    print(f"Bands created.  [3/5]")
    #  /////// Finding the candidates \\\\\\\

    candidates = LocallySensetiveHashing(banded_signatures, band_size_projection)

    print(f"Candidates are found. We found {len(candidates)} candidates [4/5]")

    # /////// Finding the matches and making the output file \\\\\\\\\

    count = 0
    filename = "dcs.txt"

    if os.path.exists(filename):
	    os.remove(filename)


    for candidate_pair in candidates:
        if time.time() - start_time > 30 *60:
                print("30 minutes passed. Terminating the algorithm early")
                break
        if Cosine_similarity(sparse_matrix,candidate_pair[0], candidate_pair[1]) > dcs_threshold:
            sorted = np.sort(candidate_pair)
            out = str(str(sorted[0])+ "," + str(sorted[1]))
            f = open(filename, "a+")
            if count != 0:
                f.write("\n")
            f.write(out)
            f.close()
            count += 1
            
    

    print(f"We found {count} similar pairs using Discrete Cosine similarity")
    print(f"Discrete Cosine similarity algoritm successfully ran. [5/5]")




#################################################################################
# Calculating the time program took

end_time = time.time()

print(f"The program took {round((end_time - start_time ) / 60)} minutes to excecute")
#################################################################################
