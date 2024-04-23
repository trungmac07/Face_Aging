import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
from FaceAgingModel import *



def load_image():
    # Open a file dialog to select an image
    file_path = filedialog.askopenfilename()
    if file_path:
        global image_org_path
        image_org_path = file_path
        # Open the image file
        img = Image.open(file_path)
        # Resize the image to fit the label
        img = img.resize((128, 128))
        global image_org 
        image_org = img
        photo = ImageTk.PhotoImage(img)
        # Set the image in the label
        image_label.config(image=photo)
        image_label.image = photo  # Keep a reference!
        age_group_label.config(text=f"No Age Group Selected")


def button_action(age_group):
    photo = None
    print(f"Button clicked for: {age_group}")
    age_group_label.config(text=f"Selected: {age_group}")
    try:
        print(image_org)
        
        image_age[age_group] = model.generate_img_UI(image_org, age_group)

        photo = ImageTk.PhotoImage(image_age[age_group])    
        image_label.config(image=photo)
        image_label.image = photo  # Keep a reference!
    except Exception as e:
        print(e)
        print(f"Unable to generate image for age group: {age_group}")
        



# Model
config = StarGANConfig()
model = FaceAgingStarGAN(config=config)

image_age = {"Juvenile" : None, "Teenager" : None, "Middle Age" : None, "Senior" : None}


# Create the main window
root = tk.Tk()
root.title("StarGAN Face Aging Model")

# Label for selected age group
age_group_label = tk.Label(root, text="No Image Is Selected")
age_group_label.pack()

# Create a frame for the image
frame = tk.Frame(root, width=128, height=128)
frame.pack(pady=20)

# Create a label to show the image
image_label = tk.Label(frame)
image_label.pack(expand=True)

# Buttons for age groups
ages = ["Juvenile", "Teenager", "Middle Age", "Senior"]
for age in ages:
    btn = tk.Button(root, text=age, command=lambda age=age: button_action(age))
    btn.pack(side=tk.LEFT, padx=10)


# Select picture button
select_btn = tk.Button(root, text="Select Picture", command=load_image)
select_btn.pack(side=tk.LEFT, padx=10)



# Run the main loop
root.mainloop()