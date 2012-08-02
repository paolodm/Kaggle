'''
Created on Aug 1, 2012

@author: Paolo
'''

from numpy import array, append
from scipy.sparse import csr_matrix, coo_matrix

# Read all of kaggle_songs.txt
# Put into a dictionary

with open('kaggle_songs.txt', 'r') as songfile:
    songs_lookup = dict()
    
    for line in songfile:
        song, ind = line.strip().split(' ')
        songs_lookup[song] = int(ind)
    pass


# Read all of kaggle_users.txt
# Put into a dictionary
with open('kaggle_users.txt', 'r') as userfile:
    users_lookup = dict()
    userInd = 1
    
    for line in userfile:
        user = line.strip()
        users_lookup[user] = userInd
        userInd = userInd + 1
    pass

# Read kaggle_visible_evaluation_triplets.txt
# for i = 0, endOfFile
#    row.add(currentSong_index)
#    col.add(currentUser_index)
#    data.add(numberplayed)
# A = coo_matrix((data, (row, col)), shape=(len(row), len(col))
# B = csr_matrix(A)

with open('kaggle_visible_evaluation_triplets.txt', 'r') as triplets:
    row = array([])
    col = array([])
    data = array([])
    
    for line in triplets:
        user, song, count = line.strip().split('\t')
        append(row, songs_lookup[song])
        append(col, users_lookup[user])
        append(data, count)
            
    A = coo_matrix((data, (row, col)), shape=(len(songs_lookup), len(users_lookup)))
    B = csr_matrix(A)
    pass


print "This is the end..."
print A.shape