#!/usr/bin/env python
# coding: utf-8

# In[1]:


from DFLIMG import DFLJPG
import glob
import numpy as np
import matplotlib.pyplot as plt
import os, sys, math
import csv
import cv2 


# In[2]:


def normalize_vector(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))

def normalize_landmarks(lm):
    c0 = normalize_vector(lm[:,0])
    c1 = normalize_vector(lm[:,1])
    return np.transpose(np.vstack([[c0], [c1]]))


# In[3]:


landmarks = []
src_filename = glob.glob('DeepFaceLab_Linux/workspace/data_src/aligned/*.jpg')
for f in src_filename:
    img = DFLJPG.load(f)
    landmarks.append(img.get_landmarks())
np.save("src_landmarks.npy", landmarks)


# In[4]:



print(len(src_filename))
print(len(landmarks))


# In[5]:



norm_landmarks = [normalize_landmarks(lm) for lm in landmarks]
#np.column_stack((NormalizeData(landmarks[0][:,0]), NormalizeData(landmarks[0][:,1])))
print(len(norm_landmarks))


# In[6]:


all_src_landmarks = norm_landmarks
dst_landmarks = []

dists = []
# This method computes the distance between two images
# by comparing their landmarks.
# It calculates the distance between two correspondig landmark's dots
# (one of the src image, and the other belonging to the dst img)
# It then sums up all the distances of a pair of images. This sum will be
# the score assigned to the face swap frame.
for ii, f in enumerate(glob.glob('DeepFaceLab_Linux/workspace/data_dst/aligned/*.jpg')):
    img = DFLJPG.load(f)
    dst_landmarks = normalize_landmarks(img.get_landmarks())
    dists_local = []
    for f_src, src_landmarks in enumerate(all_src_landmarks):
        dist = np.sum(np.linalg.norm(src_landmarks - dst_landmarks, ord=2, axis=1))
        dists_local.append((dist, f, src_filename[f_src], dst_landmarks, src_landmarks))
    # get the distance between the dst and the best src image
    dists.append((sorted(dists_local, key=lambda d: d[0]))[0])

# The Euclidean distance between all the landmarks of two images is 
# considered as the quality metric used to jugde the resulting frame

# In[7]:


# sort all samples from the best to the worst landmark pairing
dists = list(sorted(dists, key=lambda d: d[0]))
print("Primo metodo: ", dists[0][0])
dist_scores = list(zip(*dists))[0]


# In[8]:

rangeMax = np.quantile(dist_scores, 0.95)
fig = plt.figure() # Create matplotlib figure
plt.hist(dist_scores, bins=500)
plt.xlabel('Score')
plt.ylabel('Number of imgs')
plt.ylim(0, 50)
plt.axvline(rangeMax, color='red')
plt.show()


# In[9]:


print(np.quantile(dist_scores, q=0.95))

# In[10]:

img_index = 0#len(dists)-1
dst_frame_name = dists[img_index][1]
dst_frame_num = str(dists[img_index][1])[-10:-6]
src_frame_name = dists[img_index][2]
print(dst_frame_name, dst_frame_num, src_frame_name)


# In[11]:


plt.axis('equal')
plt.scatter(list(dists[img_index])[4][:, 0], -list(dists[img_index])[4][:, 1], label='src_img')


# In[12]:


plt.scatter(list(dists[img_index])[3][:, 0], -list(dists[img_index])[3][:, 1])
plt.plot(list(dists[img_index])[3][:, 0], -list(dists[img_index])[3][:, 1])
plt.axis('equal')


# In[13]:


fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.set_aspect('equal')

ax1.scatter(list(dists[img_index])[4][:, 0], -list(dists[img_index])[4][:, 1], label='src_img')
ax1.scatter(list(dists[img_index])[3][:, 0], -list(dists[img_index])[3][:, 1], c='r', label='dst_img')
plt.legend(loc='lower left');
plt.show()


# In[14]:


# This block computes the exact same evaluation of the beginning block.
# It's just for debug and check.
def point_dist(p1, p2):
    dx, dy = p2[0] - p1[0], p2[1] - p1[1]
    return math.sqrt(dx*dx + dy*dy)
src_lm = dists[img_index][3]
dst_lm = dists[img_index][4]
plt.axis('equal')
plt.scatter(src_lm[:,0], -src_lm[:,1], label='src')
plt.scatter(dst_lm[:,0], -dst_lm[:,1], label='dst')
sum_dd = 0
for x1, y1, x2, y2 in zip(src_lm[:,0], -src_lm[:,1], dst_lm[:,0], -dst_lm[:,1]):
    plt.plot([x1, x2], [y1, y2], c='r')
    dd = point_dist([x1, y1], [x2, y2])
    sum_dd += dd
plt.legend(loc='lower left');
print(sum_dd, dists[img_index][0], np.sum(np.linalg.norm(src_lm - dst_lm, ord=2, axis=1)))

print("Second metodo: ",sum_dd) 


# In[15]:


def get_frame(fname, frame_number):
    cmd = f"ffmpeg -i {fname} -vf select='eq(n\,{frame_number})' -vsync 0 frame_{frame_number}.jpg"
    os.system(cmd)
    
video_fname = '../workspace/result.mp4'
get_frame(video_fname, dst_frame_num)


# In[16]:


cmd = 'xdg-open '+dst_frame_name+'&'
os.system(cmd)
cmd = 'xdg-open frame_'+dst_frame_num+'.jpg &'
os.system(cmd)
print(cmd)
debug_name = str(dst_frame_name)[0:-12]+"_debug/"+str(dst_frame_name)[-11:-6]+'.jpg'
print(debug_name)
cmd = 'xdg-open '+debug_name+'&'
os.system(cmd)
cmd = 'xdg-open '+src_frame_name+' &'
os.system(cmd)


# In[31]:


dists_w_result = []
for i, d in enumerate(dists):
    dists_w_result.append([int(os.path.basename(dists[i][1]).split('_')[0]), d[0] < rangeMax, d[0], d[1],  d[2], "frame_"+str(dists[i][1])[-10:-6]+".jpg", d[3], d[4]])
print((dists_w_result)[0])

# In[52]:

headers = ["dst_index", "valid_face" , "landmark_distance_sum", "dst_frame_name", "src_frame_name", "result_frame_name", "landmarks"]
with open('dists.csv', 'w') as v:
    write = csv.writer(v) 
    write.writerow(headers)
    write.writerows(dists_w_result)
v.close()

