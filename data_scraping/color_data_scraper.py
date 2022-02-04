import os, cv2
from bin_detection.roipoly import RoiPoly
from matplotlib import pyplot as plt
import matplotlib
matplotlib.use('Qt5Agg')
import numpy as np

if __name__ == '__main__':

    # Training images directory from the bin_detection folder
    folder = 'bin_detection/data/training'
    #folder = 'bin_detection/data/validation'
    save_dir = 'data_scraping/data/'
    _, _, train_imgs = next(os.walk(folder))
    train_imgs = sorted(train_imgs,key=lambda x: int(os.path.splitext(x)[0]))
    
    """
        Customizable variable to decide where to start scrapping color data [1 - 60]
    """
    start_ID = 12

    for ID, filename in enumerate(train_imgs):

        # Initialize and define color data as an empty list
        color_data = []

        # Completion Status
        completion_status = False

        # Skip to the starting image
        if ID < start_ID - 1:
            continue

        # Read the training image
        img = cv2.imread(os.path.join(folder,filename))
        img_RGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_HSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        img_LAB = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        img_YCrCb = cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB)

        # While completion status is False
        while completion_status == False:

            # Display the image and use roipoly for labeling
            fig, ax = plt.subplots()
            ax.imshow(img_RGB)
            my_roi = RoiPoly(fig = fig, ax = ax, color = 'r')

            # Get the image mask
            mask = my_roi.get_mask(img_RGB)

            # Ask user whether it is a recycling-bin blue sample
            while True:
                color_type = input("Choose a color [1 - 14]:\n"\
                                        "Recycling-bin Blue: 1\n"\
                                        "Skyblue           : 2\n"\
                                        "Waterblue         : 3\n"\
                                        "Purple            : 4\n"\
                                        "Dark Blue         : 5\n"\
                                        "Blue              : 6\n"\
                                        "Light Blue        : 7\n"\
                                        "Red               : 8\n"\
                                        "Brown             : 9\n"\
                                        "Tan               : 10\n"\
                                        "Yellow            : 11\n"\
                                        "Green             : 12\n"\
                                        "Gray              : 13\n"\
                                        "Black             : 14\n"\
                                        "White             : 15\n")
                # Convert string to integer                        
                color_type = int(color_type)
                # Make sure the input is in between the color selection range
                if 1 <= color_type <= 15:
                    break
                # Ask user to re-select appropriate color number
                print("\nError 404: Color not found. Please try again.\n")

            # Extract color data from the mask
            for height in range(mask.shape[0]):
                for width in range(mask.shape[1]):
                    if mask[height, width]:
                        color_data.append([img_RGB[height, width, 0], img_RGB[height, width, 1], img_RGB[height, width, 2],
                                           img_HSV[height, width, 0], img_HSV[height, width, 1], img_HSV[height, width, 2],
                                           img_LAB[height, width, 0], img_LAB[height, width, 1], img_LAB[height, width, 2], 
                                           img_YCrCb[height, width, 0], img_YCrCb[height, width, 1], img_YCrCb[height, width, 2], 
                                           color_type])

            # Ask user's completion status
            while True:
                user_status = input("\nDone extracting data from this image?[Y/N]: ")
                if user_status not in ['Y','y','N','n']:
                    print("\nError. Please input valid response.\n")
                    continue
                if user_status in ['Y','y']:
                    completion_status = True
                    break
                if user_status in ['N','n']:
                    break
        
        # Convert list to array
        color_data = np.rint(np.array(color_data))

        # Save as csv file to "bin_detection/data/training'"
        np.savetxt(save_dir + filename.replace("jpg", "csv"), color_data, delimiter=",")