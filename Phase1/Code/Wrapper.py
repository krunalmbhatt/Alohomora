#!/usr/bin/env python3

"""
RBE/CS549 Spring 2022: Computer Vision
Homework 0: Alohomora: Phase 1 Starter Code

Colab file can be found at:
	https://colab.research.google.com/drive/1FUByhYCYAfpl8J9VxMQ1DcfITpY8qgsF

Author(s): 
Prof. Nitin J. Sanket (nsanket@wpi.edu), Lening Li (lli4@wpi.edu), Gejji, Vaishnavi Vivek (vgejji@wpi.edu)
Robotics Engineering Department,
Worcester Polytechnic Institute

Code adapted from CMSC733 at the University of Maryland, College Park.
"""

"""
Coded by       : Krunal M. Bhatt
ID             : 901018587
Email          : kmbhatt@wpi.edu
Date           : 1/7/2024 
"""

#####################
# Code starts here: #
#####################
#Importing the libraries
import glob
import os
import numpy as np
import cv2
from scipy.signal import convolve2d
from skimage.transform import rotate
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


def gaussian_2d(size, sigma):
	x, y = np.meshgrid(np.arange(-size // 2, size // 2 ), np.arange(-size // 2, size // 2 )) #2D array created
	variance = sigma ** 2 #Variance calculated from sigma 
	k = x ** 2 + y ** 2   #Numerator for the exponential term
	output_gaussian_2d = np.exp(-k / (2*variance)) / (2*np.pi* variance)  #calculating the gaussian
	return output_gaussian_2d

def pad_image(image, padding_height, padding_width):
	image_height, image_width = image.shape				#Image size calculated
	padded_image = np.zeros((image_height + 2 * padding_height, image_width + 2 * padding_width), dtype=image.dtype)  #Padding the image
	padded_image[padding_height:padding_height+image_height, padding_width:padding_width+image_width] = image  #Padding the image
	return padded_image

def convolution(image, kernel):
	kernel_height, kernel_width = kernel.shape  #Kernel size calculated
	image_height, image_width = image.shape     #Image size calculated
	output = np.zeros((image_height, image_width), dtype=image.dtype)   #Output image created
	padding_height = kernel_height // 2   # Padding height calculated
	padding_width = kernel_width // 2     # Padding width calculated
	
	padded_image = pad_image(image, padding_height, padding_width)  #Padding the image
	
	for i in range(image_height):        #Convolution operation
		for j in range(image_width):     
			output[i, j] = np.sum(padded_image[i:i+kernel_height, j:j+kernel_width] * kernel)  
	return output

def laplacian_of_gaussian_2d(n, sigma):
	size = int((n-1)/2)	# Calculate the radius of the kernel
	var = sigma**2 
	arr = np.asarray([[x**2 + y**2 for x in range(-size, size+1)] for y in range(-size, size+1)]) # Create a 2D array same size as the kernel
	gaussian = (1/np.sqrt(2*np.pi*var)) * np.exp(-arr/(2*var)) # Calculate the Gaussian part of the LoG
	laplacian = (arr - var) / (var**2) # Calculate the Laplacian part
	output = gaussian * laplacian #LoG
	return output

def gaussian_1d(sigma, mean, x, order):
	x = np.array(x) - mean  # calculating the numerator for the exponential term
	variance = sigma ** 2   # calculating the variance
	output_gaussian_1d = np.exp(-x ** 2 / (2*variance)) / (np.sqrt(2*np.pi*variance))  # calculating the gaussian
	if order == 1:  # first derivative
		output_gaussian_1d = -x * output_gaussian_1d / variance
	elif order == 2:  # second derivative
		output_gaussian_1d = (x ** 2 - variance) * output_gaussian_1d / variance ** 2
	return output_gaussian_1d 

def create_binary_image(img, bin_value):
	binary_img = np.zeros(img.shape, dtype=img.dtype)  # Binary image with the same shape as the input image
	for i in range(img.shape[0]):  
		for j in range(img.shape[1]):  
			if img[i, j] == bin_value:  # Check if the pixel value is equal to the specified bin value
				binary_img[i, j] = 1  # Set the corresponding pixel in the binary image to 1
	return binary_img  # Return the binary image

# Function to calculate the gradient map
def gradient(image_map, num_bins, mask_left, mask_right):
	gradient_map = np.zeros((image_map.shape[0], image_map.shape[1], 12))  # Initialize gradient map with zeros
	for m in range(0, 12):  
		chi_squared = np.zeros(image_map.shape)  # Initialize chi_squared array with zeros
		for i in range(1, num_bins):  
			binary_image = create_binary_image(image_map, i)  # Create binary image 
			conv_left = cv2.filter2D(binary_image, -1, mask_left[m])  # Convolve binary image with left mask
			conv_right = cv2.filter2D(binary_image,-1, mask_right[m])  # Convolve binary image with right mask
			denominator = conv_left + conv_right + 0.0001 
			chi_squared += ((conv_left - conv_right) ** 2) / denominator  # Calculate chi-squared value
		gradient_map[:, :, m] = chi_squared  
	return gradient_map

def filter_2d_xy(scale, x, y, pts, sup):
	gx = gaussian_1d(3*scale, 0,pts[0,...], x) 	#1D alogng x-axis
	gy = gaussian_1d(scale, 0,pts[1,...], y)    #1D along y-axis
	img = gx*gy		#2D filter
	img = np.reshape(img,(sup,sup))
	return img

#Filter Bank for directed DoG filter
def DoG_bank(sigma, orientation):
	sobel_x = np.asarray([[-1,0,1], [-2,0,2],[-1,0,1]])
	sobel_y = np.asarray([[-1,-2,-1],[0,0,0],[1,2,1]])	
	DoG_filters = []
	len_sigma = len(sigma)
	o = orientation.size
	rows = []
	plt.figure(figsize=(15,2))
	for i in range(0, len_sigma):
		filter_new_x  = cv2.filter2D(gaussian_2d(10, sigma[i]),-1, sobel_x)
		filter_new_y  = cv2.filter2D(gaussian_2d(10, sigma[i]),-1, sobel_y)
		filter_new= filter_new_x + filter_new_y
		cols = []
		for j in range(0, o):
			filt_rot = rotate(filter_new, orientation[j])
			DoG_filters.append(filt_rot)
			plt.subplot(len_sigma, o, o*(i)+j+1)
			plt.axis('off')
			plt.imshow(DoG_filters[o*(i)+j], cmap='gray')
			# Normalize the filter values to 0-255
			norm_filter = cv2.normalize(filt_rot, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
			cols.append(norm_filter)
		rows.append(np.hstack(cols))
	img = np.vstack(rows)
	plt.show()
	return img

#Filter Bank for Leung-Malik filter
def lm_bank():
	size_of_filter = 49  # Size of the filter
	scalex = np.array([1, np.sqrt(2), 2])  # Scaling factors
	#scalex  = np.sqrt(2) * np.array([1,2,3])
	num_orient = 6  # Number of orientations
	num_rot_inv = 12  # Number of rotated invariant filters

	number_of_bar = len(scalex) * num_orient  # Number of bar filters
	num_of_edge = len(scalex) * num_orient  # Number of edge filters
	tot_f = number_of_bar + num_of_edge + num_rot_inv  # Total number of filters
	LMF = np.zeros([size_of_filter, size_of_filter, tot_f])  # Filter bank
	hsup = (size_of_filter - 1) / 2  # Half of the filter size
	x, y = np.meshgrid(np.arange(-hsup, hsup + 1), np.arange(-hsup, hsup + 1))

	orginal_pts = [x.flatten(), y.flatten()]  # Flatten the coordinates
	orginal_pts = np.array(orginal_pts)

	count = 0
	for scale in range(len(scalex)):
		for orient in range(num_orient):
			angle = (np.pi * orient) / num_orient  # Calculate the angle
			c = np.cos(angle)  # Cosine of the angle
			s = np.sin(angle)  # Sine of the angle
			rotation_pts = [[c, s], [s, -c]]  # Rotation matrix
			rotation_pts = np.array(rotation_pts)  # Convert to numpy array
			rotation_pts = np.dot(rotation_pts, orginal_pts)  # Apply rotation to the coordinates
			LMF[:,:,count] = filter_2d_xy(scalex[scale], 0, 1, rotation_pts, size_of_filter)  # Create bar filter
			count = count + 1  # Increment count by 1 after each filter is created

	count = num_of_edge
	for scale in range(len(scalex)):
		for orient in range(num_orient):
			angle = (np.pi * orient) / num_orient
			c = np.cos(angle) 
			s = np.sin(angle) 
			rotation_pts = [[c, s], [s, -c]]  
			rotation_pts = np.array(rotation_pts)  
			rotation_pts = np.dot(rotation_pts, orginal_pts)  
			LMF[:, :, count] = filter_2d_xy(scalex[scale], 0, 2, rotation_pts, size_of_filter)  # Create edge filter
			count = count + 1  # Increment count by 1 after each filter is created

	count = number_of_bar + num_of_edge

	scales = np.array([np.sqrt(2), 2, 2*np.sqrt(2), 4])	# Scaling factors for Gaussian and Laplacian filters  
	#scales = np.sqrt(2) * np.array([1,2,3,4])

	for i in range(len(scales)):
		LMF[:, :, count] = laplacian_of_gaussian_2d(size_of_filter, scales[i])  # Create Laplacian of Gaussian filter
		count = count + 1

	for i in range(len(scales)):
			LMF[:, :, count] = laplacian_of_gaussian_2d(size_of_filter, 3 * scales[i])  # Create Laplacian of Gaussian filter with larger scale
			count = count + 1
	
	for i in range(len(scales)):
		LMF[:, :, count] = gaussian_2d(size_of_filter, scales[i])  # Create Gaussian filter
		count = count + 1

	return LMF

# Function to generate Gabor filter
def generate_gabor(size, omega, theta, sigma):
	radius = (int(size[0]/2.0), int(size[1]/2.0)) # Calculate the radius of the filter
	[x, y] = np.meshgrid(range(-radius[0], radius[0]+1), range(-radius[1], radius[1]+1))
	x_rotated = x * np.cos(theta) + y * np.sin(theta)  # Rotate the coordinates based on the given theta angle
	y_rotated = -x * np.sin(theta) + y * np.cos(theta)
	gaussian = 1 / (2 * np.pi * sigma**2) * np.exp(-(x_rotated**2 + y_rotated**2) / (2 * sigma**2))
	sine = np.sin(omega * x_rotated)  # Calculate the sin component
	gabor_filter = gaussian * sine  # Combine the Gauss and sin components to get the Gabor filter
	
	return gabor_filter

# Function to generate Gabor filter bank
def gabor_bank():
	theta_range = np.arange(0, np.pi, np.pi/8)  # Define the range of theta
	omega_range = np.arange(0.2,0.6,0.1)  # Define the range of omega
	sigma_range = np.arange(15,25,5)  # Define the range of sigma
	params = [(theta, omega, sigma) for sigma in sigma_range for omega in omega_range for theta in theta_range]  # Generate all possible combinations of theta, omega and sigma
	filter_bank = [] 
	gabor_params = []
	
	for (theta, omega, sigma) in params:  # Iterate over each combination of theta, omega and sigma
		gabor_param = {'omega': omega, 'theta': theta, 'sigma': sigma, 'sz': (128, 128)}  # Define the Gabor filter parameters
		gabor_filter = generate_gabor((49,49), omega, theta, sigma)  # Generate the Gabor filter 
		filter_bank.append(gabor_filter)
		gabor_params.append(gabor_param)      
	
	num_filters = len(filter_bank)
	num_rows = int(np.ceil(num_filters / 8.0))  # Calculate the number of rows needed
	
	for i in range(num_filters):
		plt.subplot(num_rows, 8, i+1)
		plt.axis('off')
		plt.imshow(filter_bank[i], cmap='gray')
	plt.savefig('Gabor.png')
	plt.show()
		 
	return filter_bank, gabor_params  

def half_disk(radius):
	halfdisk = np.zeros((radius*2, radius*2))  # Init zero-filled array with shape (radius*2, radius*2)
	for i in range(radius):  
		for j in range(2*radius):  
			if (i-radius)**2 + (j-radius)**2 < radius**2:  # Check if distance from center is less than radius squared
				halfdisk[i, j] = 1  # Set the value at index (i, j) to 1
	return halfdisk  

def plot(img,cmap=None):
	plt.imshow(img,cmap)
	plt.axis('off')
	plt.show()




################
#main function #
################



def main():
	
	"""
	Generate Difference of Gaussian Filter Bank: (DoG)
	Display all the filters in this filter bank and save image as DoG.png,
	use command "cv2.imwrite(...)"
	"""
	sigma_values = [1,3]   #Sigma values for the Gaussian filter
	orientation_values = np.arange(0, 360,360/16)  #Orientation values for the Gaussian filter
	DoG_filters = DoG_bank(sigma_values, orientation_values)  #DoG filter bank
	cv2.imwrite('DoG.png', DoG_filters)

	"""
	Generate Leung-Malik Filter Bank: (LM)
	Display all the filters in this filter bank and save image as LM.png,
	use command "cv2.imwrite(...)"
	"""
	LM_filters = lm_bank()
	plt.figure(figsize=(12,4))
	for i in range(0,48):
		plt.subplot(4,12,i+1)
		plt.axis('off')
		plt.imshow(LM_filters[:,:,i], cmap = 'gray')
	plt.savefig('LM.png')
	plt.show()
	
	"""
	Generate Gabor Filter Bank: (Gabor)
	Display all the filters in this filter bank and save image as Gabor.png,
	use command "cv2.imwrite(...)"
	"""
	Gabor_filters = gabor_bank()

	filter_bank = []
	count = 0
	for i in range(0,len(DoG_filters)): 
		if len(DoG_filters[i].shape) == 2:  # Check if the filter is a 2D array
			filter_bank.append(DoG_filters[i])  
			count = count+1

	for i in range(0,48):
		if len(LM_filters[:,:,i].shape) == 2:  # Check if the filter is a 2D array
			filter_bank.append(LM_filters[:,:,i])
			count = count+1

	for i in range(len(Gabor_filters)):
		if len(Gabor_filters[i]) == 2:  # Check if the filter is a 2D array
			filter_bank.append(Gabor_filters[i])
			count = count+1
		
	"""
	Generate Half-disk masks
	Display all the Half-disk masks and save image as HDMasks.png,
	use command "cv2.imwrite(...)"
	"""
	orients = np.arange(0,360,360/4) #Orientation values for the Half-disk masks
	# print(orients)
	radii = np.asarray([5,10,15])  #Radius values for the Half-disk masks
	mask_l = []
	mask_r = []
	radius_size = radii.size 
	orient_size = orients.size
	# print(radius_size, orient_size)
	for i in range(0,radii.size):
		hd = half_disk(radii[i])
		for m in range(0,orients.size):
			mask_1 = rotate(hd,orients[m])
			mask_l.append(mask_1)
			mask_2 = rotate(mask_1,180) 
			mask_r.append(mask_2)
			
	plt.figure(figsize=(8, 8))
	for j in range(radius_size):
		for k in range(orient_size):
			plt.subplot(radius_size, 2*orient_size, 2*(orient_size*j+k)+1) 
			plt.axis('off') 	
			plt.imshow(mask_l[orient_size*j+k], cmap='gray')
			plt.subplot(radius_size, 2*orient_size, 2*(orient_size*j+k)+2)
			plt.axis('off')
			plt.imshow(mask_r[orient_size*j+k], cmap='gray')
	plt.savefig('HDMasks.png')
	plt.show()

	os.chdir("C:\D_FILES\Study\MS\Sem2\Computer_vision\homework\HW_0\YourDirectoryID_hw0\Phase1\BSDS500\Images") #Change directory to BSDS500/Images
	cwd = os.getcwd() #Get current working directory
	#print(cwd)
	store_img = [] #List to store image
	for img_path in sorted(glob.glob("*.jpg")): #Read all jpg images in the folder
		n= cv2.imread(img_path) #Read image
		store_img.append(n) 
	os.chdir("../../Code")


	img = 9

	plot(cv2.cvtColor(store_img[img],cv2.COLOR_BGR2RGB))


	"""
	Generate Texton Map
	Filter image using oriented gaussian filter bank
	"""
	texton_map=cv2.cvtColor(store_img[img],cv2.COLOR_BGR2GRAY)
	data = np.zeros((texton_map.size,len(filter_bank)))
	print("Starting texton computation...")
	for i in range(0,len(filter_bank)):
		temp = cv2.filter2D(texton_map, -1, filter_bank[i])
		temp = temp.reshape((1,texton_map.size))
		data[:,i] = temp
	print("Finished texton computation.")

	"""
	Generate texture ID's using K-means clustering
	Display texton map and save image as TextonMap_ImageName.png,
	use command "cv2.imwrite('...)"
	"""
	print("Starting K-means clustering...")
	k_means = KMeans(n_clusters=64, n_init=4)
	k_means.fit(data)
	labels = k_means.labels_
	print("Finished K-means clustering.")
	texton_map = np.reshape(labels,(texton_map.shape))

	plot(texton_map)
	plt.imsave("TextonMap_"+str(img)+".png",texton_map)
	plt.show()

	"""
	Generate Texton Gradient (Tg)
	Perform Chi-square calculation on Texton Map
	Display Tg and save image as Tg_ImageName.png,
	use command "cv2.imwrite(...)"
	"""
	print("Starting Tg computation...")
	tg = gradient(texton_map,64,mask_l,mask_r)
	tgm = np.mean(tg, axis=2)
	plot(tgm)
	plt.imsave("Tg_"+str(img)+".png",tgm)
	print("Finished Tg computation.")
	plt.show()


	"""
	Generate Brightness Map
	Perform brightness binning 
	"""
	print("Starting brightness map computation...")
	gray_img = cv2.cvtColor(store_img[img],cv2.COLOR_BGR2GRAY)  #Convert to grayscale
	r = gray_img.reshape((gray_img.shape[0]*gray_img.shape[1]),1)   #Reshape to 1D array
	k_means = KMeans(n_clusters=16, random_state=4) #K-means clustering
	k_means.fit(r) 
	labels = k_means.labels_ 
	brightness_map = np.reshape(labels,(gray_img.shape[0],gray_img.shape[1]))  #Reshape to 2D array
	low = np.min(brightness_map) #Normalize the image to 0-255
	high = np.max(brightness_map) 
	brightness_map_f = 255*(brightness_map-low)/float((high-low)) #Convert to 8-bit image	
	
	plot(brightness_map_f) #Display image
	print("Finished brightness map computation.")
	plt.imsave("BrightnessMap_"+str(img)+".png",brightness_map_f)
	plt.show()


	"""
	Generate Brightness Gradient (Bg)
	Perform Chi-square calculation on Brightness Map
	Display Bg and save image as Bg_ImageName.png,
	use command "cv2.imwrite(...)"
	"""
	print("Starting Bg computation...")
	bg = gradient(brightness_map,16,mask_l,mask_r)
	bgm = np.mean(bg, axis=2) # mean
	plot(bgm)
	cv2.imwrite("Bg_"+str(img)+".png",bgm)
	print("Finished Bg computation.")
	plt.show()

	"""
	Generate Color Map
	Perform color binning or clustering
	"""
	print("Starting color map computation...")
	new_ = store_img[img]
	m = new_.reshape((new_.shape[0]*new_.shape[1]),3)
	k_means = KMeans(n_clusters=16, random_state=4)
	k_means.fit(m)
	labels = k_means.labels_
	color_map = np.reshape(labels,(new_.shape[0],new_.shape[1]))
	print("Finished color map computation.")
	plot(color_map)
	plt.imsave("ColorMap_"+str(img)+".png",color_map)
	plt.show()

	"""
	Generate Color Gradient (Cg)
	Perform Chi-square calculation on Color Map
	Display Cg and save image as Cg_ImageName.png,
	use command "cv2.imwrite(...)"
	"""
	print("Starting Cg computation...")
	cg = gradient(color_map,16,mask_l,mask_r)
	cgm = np.mean(cg, axis=2)
	plot(cgm)
	plt.imsave("Cg_"+str(img)+".png",cgm)
	print("Finished Cg computation.")
	plt.show()


	"""
	Read Canny Baseline
	use command "cv2.imread(...)"
	"""
	
	os.chdir("../BSDS500/CannyBaseline")
	cwd = os.getcwd()
	canny_gray = cv2.imread(str(img)+".png")
	plot(canny_gray)
	os.chdir("../../Code")

	
	"""
	Read Sobel Baseline
	use command "cv2.imread(...)"
	"""
	os.chdir("../BSDS500/SobelBaseline")
	sobel_gray = cv2.imread(str(img)+".png")
	plot(sobel_gray)
	os.chdir("../../Code")


	"""
	Combine responses to get pb-lite output
	Display PbLite and save image as PbLite_ImageName.png
	use command "cv2.imwrite(...)"
	"""
	
	print("Starting PbLite computation...")
	sobel_gray = cv2.cvtColor(sobel_gray,cv2.COLOR_BGR2GRAY)
	canny_gray = cv2.cvtColor(canny_gray,cv2.COLOR_BGR2GRAY)
	w = 0.5
	avg= ((tgm+bgm+cgm)/3)
	cs = (w*canny_gray+(1-w)*sobel_gray)
	pb = np.multiply(avg,cs) #PbLite computation with hadamard product
	plot(pb,cmap='gray')
	plt.imsave("PbLite_"+str(img)+".png",pb,cmap='gray')
	print("Finished PbLite computation.")
	plt.show()
	
	
if __name__ == '__main__':
	main()
 


