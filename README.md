# ECE276A-Robotics-Project-1-WI22
## Blue Recycling Bin Detection using Multiclass Logistic Regression

### How to Run "Pixel_Classification"?
1.) Open up the "test_pixel_classifier.py" from the "pixel_classification" folder.
2.) Run the "test_pixel_classifier.py" code to perform pixel classification.
3.) It will output the "Precision" score.

### How to Run "Bin_Detection"?
1.) Open up the "test_bin_detector.py" from the "bin_detection" folder.
2.) Uncomment Line 75 and 77 in the "test_bin_detector.py" 
3.) Open up the "bin_detector.py" from the "bin_detection" folder.
4.) Uncomment the entire "def draw_bounding_boxes(self, mask_img, boxes, rgb_img)" function in the "bin_detector.py".
5.) Uncomment Line 12-14 in the "bin_detector.py".
6.) Run the "test_bin_detector.py" to perform pixel classification, image segmentation, display mask images and original images with bounding boxes.
7.) It will output the segmented image, RGB image with bounding boxes, and the bounding boxes list.

### How to Run the Autograder loaclly?
1.) Open up the "run_tests.py"
2.) Run the script.
3.) Wait for maximum 600 seconds. 
4.) The score will be released for the pixel classification and bin detection.

### Additional Folders and Files Added to Complete the Project.
**1.) "data_scraping" folder** (Note: Please save the data from "data_scraping/data/" before running. Running it will override previous data)
a.) "color_data_scraper.py" is used to collect a portion of pixel values from the training images and to hand label them manually.
    i.) To run the script, open up the "color_data_scraper.py" from the "data_scraping" folder. 
    ii.) In line 20, there is a defined variable called "start_ID" that you can modify from 1 to 60. It will starts at the training images that you wanted to collect data from. For example, if "start_ID" is 20, it will start at "0020.jpg" to collect data. Then, run the script.
    iii.) Select the Region of Interest (ROI) by left-clicking.
    iv.) To exit the task, just right-click the image.
    v.) Then, select a number from 1 to 15, which you think is the color.
    vi.) Type "Y" to continue select ROI in the same image, and repeat steps iii.) to v.). If not, type "N" to move to the next image.
    vii.) The process is repeated from iii.) to vi.) until the last image, which is "0060.jpg"
    viii.) Data will be saved at "data_scraping/data/".
b.) "test_data_loader.py" is used to compiled data from "data_scraping/data/"  and save it at "data_scraping/compiled_color_data/" as a .csv file.
    i.) To run the script, open up the "test_data_loader.py" from the "data_scraping" folder. 
    ii.) Run the script.
    iii.) Data from from "data_scraping/data/" is complied and saved to "data_scraping/compiled_color_data/" as a .csv file.
