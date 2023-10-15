import os

# Set the path to your main folder
# main_folder_path = "D:\\dahyun\\Photobook"
main_folder_path = "D:\\dahyun\\Photobook\\t"

# Create a dictionary to store the count of each file extension
file_extension_count = {}

# Create a list to store the .png file names
png_file_names = []

# Walk through all the directories and subdirectories in the main folder
for root, dirs, files in os.walk(main_folder_path):
    # Loop through all the files in the current directory
    for file in files:
        # Get the file extension
        file_extension = os.path.splitext(file)[1]
        # If the file extension is not in the dictionary, add it with a count of 1
        if file_extension not in file_extension_count:
            file_extension_count[file_extension] = 1
        # If the file extension is already in the dictionary, increment its count by 1
        else:
            file_extension_count[file_extension] += 1
        # If the file has a .png extension, add its name to the list
        if file_extension == ".png":
            png_file_names.append(file)

# Print the count of each file extension
for file_extension, count in file_extension_count.items():
    print(f"{file_extension}: {count}")

# Print the list of .png file names
# print("List of all .png files:")
# for file_name in png_file_names:
#     print(file_name)
