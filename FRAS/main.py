import os  # accessing the os functions
import check_camera
import capture_image
import train_image
import facial_analysis
import recognize


# creating the title bar function

def title_bar():
    os.system('cls')  # for windows

    # title of the program

    print("\t**********************************************")
    print("\t***** Face Recognition Attendance System *****")
    print("\t**********************************************")


# creating the user main menu function

def main_menu():
    title_bar()
    print()
    print(10 * "*", "WELCOME MENU", 10 * "*")
    print("[1] Check Camera")
    print("[2] Capture Faces")
    print("[3] Train Images")
    print("[4] Recognize & Attendance")
    print("[5] Auto Mail")
    print("[6] Facial Analysis")
    print("[X] Quit")

    while True:
        choice = str(input("Enter Choice: "))
        
        if choice == "1":
            check_cam()
            break
        elif choice == "2":
            capture_faces()
            break
        elif choice == "3":
            train_images()
            break
        elif choice == "4":
            recognize_faces()
            break
        elif choice == "5":
            os.system("py automail.py")
            break
        elif choice == "6":
            facial_analysis_driver()
            break
        elif choice == "X":
            print("Thank You")
            break
        else:
            print("Invalid Choice.")
            main_menu()
    exit


# ---------------------------------------------------------
# calling the camera test function from check camera.py file

def check_cam():
    check_camera.camera()
    key = input("Enter any key to return main menu")
    main_menu()


# --------------------------------------------------------------
# calling the take image function form capture image.py file

def capture_faces():
    capture_image.take_images()
    key = input("Enter any key to return main menu")
    main_menu()


# -----------------------------------------------------------------
# calling the train images from train_images.py file

def train_images():
    train_image.train_images()
    key = input("Enter any key to return main menu")
    main_menu()


# --------------------------------------------------------------------
# calling the recognize_attendance from recognize.py file

def recognize_faces():
    recognize.recognize_attendence()
    key = input("Enter any key to return main menu")
    main_menu()

# --------------------------------------------------------------------
# calling analysis for facial analysis (age, gender, and sentiment) from facial_analysis.py

def facial_analysis_driver():
    facial_analysis.run_analysis()
    key = input("Enter any key to return to main menu")
    main_menu()

# ---------------main driver ------------------
main_menu()
