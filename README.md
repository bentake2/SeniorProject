Segmentation Project



Task: develop an Android application that performs automatic 
segmentation of an image that is selected and loaded by the user from the storage memory of the 
android phone using Android Studio, python and pytorch

![image](https://user-images.githubusercontent.com/62612740/156855945-d7962ca0-d785-43ef-8c75-4a09781719f3.png)

Overview/Breakdown: Over the next eight weeks we will develop, bulid, test, and train our models and application to perfom image segmentation in which  a digital image is broken down into various subgroups called Image segments. we will also be giving bi-weekly updates on the process


Step to run the app with the model

Download android studio https://developer.android.com/studio/
Go to https://github.com/bentake2/SeniorProject/tree/John-Michael-Kuczynski 
Hit Code => Download ZIP
Run the code and download the ptl file https://colab.research.google.com/drive/1gwYMLYvijCRkHb1PCgg9KrsHHuMxXK25?usp=sharing#scrollTo=woGO10XFORRm
Unzip the project 
File => Open => click the app
Put the ptl model in the assets folder



Tools => Device Manager
Virtual Device tab => Create Device => Pixel 5
Hit the folder in the action tab of the device
sdcard => Download (put your image there)
close the device
In the actions tab => hit the down arrow => Cold  Boot Now
With your virtual device selected in the tool bar hit the green arrow button/run app
Click load image => Select your image you want to segment
Click the Segment button next
