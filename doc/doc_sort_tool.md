### **Sort tool**:

`blur` places most blurred faces at end of folder

`hist` groups images by similar content

`hist-dissim` group images by dissimilarity, placing the most similar to each other images to end.

`hist-blur` sort by a function of similar content and blur 

`face-pitch` sort by face pitch direction

`face-yaw` sort by face yaw direction

`brightness` 

`hue`

`black` places images which contains black area at end of folder. Useful to get rid of src faces which cutted by screen.

`final` sorts by yaw, blur, and hist, and leaves best 1500-1700 images.

### **Extraction Workflow**

Extraction is rarely perfect and a final pass by human eyes is typically required. This is the best way to remove unmatched or misaligned faces quickly. The sort tool included in the project greatly reduces the time and effort required to clean large sets. Like pictures will be grouped together and false positives can be quickly be identified.

Suggested sort workflow for gathering cleaning face sets from very large image pools:

1) `black` -> then delete faces with black edges at end of folder
2) `blur` -> then delete blurred faces at end of folder, use your judgement
3) `hist` -> then delete groups of mismatched faces, leaving only target face
4) `final` -> then delete faces blocked by obstructions (hands, hair, etc)

Suggested sort workflow for preparing and cleaning face sets from very large image pools:

1) Manually delete unsorted aligned groups of images what you can to delete, ignore instances of your target face for now.
2) `hist` -> then delete groups of mismatched faces, leaving only target face. 
