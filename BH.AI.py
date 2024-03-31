from tkinter import *
import tkinter.scrolledtext as tkst
from tkinter import messagebox, ttk
from datetime import datetime
from PIL import ImageTk, Image
import tkinter as tk
import os
import customtkinter
from typing import Any
customtkinter.set_appearance_mode('dark')
customtkinter.set_default_color_theme('green')

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer
from textblob import TextBlob, Word
import nltk
import random
from nltk import wordnet
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
nltk.download('omw-1.4')
from sklearn.svm import LinearSVC

from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

from sklearn.model_selection import train_test_split
import json

column_names = ['Input','Intent']
df = pd.read_csv("C:/Users/NAMAHIGA/Desktop/BH.AI/emos.csv", names = column_names)

print(df.head())
print(df.describe())
print(df.info())

unique, counts = np.unique(df['Intent'], return_counts=True)
print(dict(zip(unique, counts)))

def lemmatize_with_postag(sentence):
    sent = TextBlob(sentence)
    tag_dict = {"J": 'a', 
                "N": 'n', 
                "V": 'v', 
                "R": 'r'}
    words_and_tags = [(w, tag_dict.get(pos[0], 'n')) for w, pos in sent.tags]    
    lemmatized_list = [wd.lemmatize(tag) for wd, tag in words_and_tags]
#     if 'no' or 'not' in sentence:
#         lemmatized_list.append('not')
    return " ".join(lemmatized_list)
print(df["Input"].loc[2])
df["Input"] = df["Input"].apply(lemmatize_with_postag)
print(df["Input"].loc[2])

vectorizer = TfidfVectorizer(
    stop_words="english",
)

X_train, X_test, y_train, y_test = train_test_split(df['Input'], df['Intent'], test_size=0.2, random_state=1000)
print(type(X_test))
svm = LinearSVC()
X_train = vectorizer.fit_transform(X_train)
X_test = vectorizer.transform(X_test)
_ = svm.fit(X_train, y_train)
y_pred = svm.predict(X_test)
print(classification_report(y_test, y_pred))

def predict_intent(msg):
    a = vectorizer.transform([msg])
    a1 = svm.predict(a)
    final_intent = a1[0]
    return final_intent
	
data_file = open("C:/Users/NAMAHIGA/Desktop/BH.AI/intents_1.json").read()
intents = json.loads(data_file)

def get_response(msg,a1):
    for intent in intents['intents']: 
        if intent['tag'] != a1:
            if len(intent['patterns']) > 1:
                for pattern in intent['patterns']:
                    words = nltk.word_tokenize(pattern)
                    print(words)
                    words_1 = nltk.word_tokenize(a1)
                    print(words_1)
                    for w in words:
                        if w in " ".join(words_1):
                            result = random.choice(intent['responses'])
                            return result
        else:
            if intent['tag'] == a1:
                l = []
                result = random.choice(intent['responses']) 
                l.append(result)
                result1 = random.choice(intent['context'])
                l.append(result1)
            return l
#-----------------------------------------------------------------------------

class Application(customtkinter.CTk):
    def __init__(self, *args, **kwargs):
        customtkinter.CTk.__init__(self, *args, **kwargs)
        
        self.state('zoomed')
        self.update()
        self.maxsize(-1,self.winfo_height())
        self.state('normal')
        self.wip=False
        self.lastgeom=None
        self.bind('<Configure>',self.adopt)
        #without this after disabling and reenabling resizing in zoomed state
        #the window would get back to normal state but not to it's prior size
        #so this behavior is secured here
        self.bind('<x>',self.locksize)
        
        # set the window title
        self.title("BH.AI")
        # create a stack of frames for the pages
        self.frames: list[Any] = []

        # create the main page
        self.main_page = MainPage(self)
        self.frames.append(self.main_page)
        self.main_page.grid(row=0, column=0, sticky="nsew")
        
        # Create a photoimage object of the image in the path
        # imagebh = Image.open("C:/Users/NAMAHIGA/Desktop/BH.AI/BH.AI.png")
        # test = customtkinter.CTkImage(imagebh)

        # label1 = customtkinter.CTkLabel(image=test)
        # label1 = test

        # Position image
        # label1.grid(row=5,column=5)
        
        # create the first page
        self.Jnl = Jnl(self)
        self.frames.append(self.Jnl)
        self.Jnl.grid(row=0, column=0)
        
        
        # create the second page
        self.chat = Chatter(self)
        self.frames.append(self.chat)
        self.chat.grid(row=1, column=0)
        
        
        # show the first page
        self.show_frame(self.main_page)
    def adopt(self,event):
        if not self.wip:
            self.wip=True
            if self.state()=='zoomed' and not self.lastgeom:
                self.state('normal')
                self.update()
                self.lastgeom=self.geometry()
                self.state('zoomed')
            elif self.state()=='normal' and self.lastgeom:
                self.geometry(self.lastgeom)
                self.lastgeom=None
            self.wip=False
    def locksize(self,event):
        if self.resizable()[0]: self.resizable(False,False)
        else: self.resizable(True,True)
    def show_frame(self, frame):
        # hide all frames except the one to be shown
        for f in self.frames:
            if f == frame:
                f.tkraise()
            else:
                f.grid_forget()
        
        # show the selected frame
        frame.grid(row=0, column=0)
        
#-------------------------------------------------
class MainPage(customtkinter.CTkFrame):
    def __init__(self, master):
        customtkinter.CTkFrame.__init__(self, master)
        
        # create some widgets for the main page
        self.label = customtkinter.CTkLabel(self,text="BH.AI",width=1000,height=500,corner_radius=10,font=("Arial", 50))
        self.label.grid(row=0, column=0)
        
        
        self.button1 = customtkinter.CTkButton(self, text="Journal", command=self.go_to_page1,height=50, width=150)
        self.button1.grid(row=1, column=0,pady=5)
        
        
        self.button2 = customtkinter.CTkButton(self, text="Chat with Bhai", command=self.go_to_page2,height=50, width=150)
        self.button2.grid(row=2, column=0,pady=5)
        

    def go_to_page1(self):
        # navigate to page 1
        self.master.show_frame(self.master.Jnl)
    
    def go_to_page2(self):
        # navigate to page 2
        self.master.show_frame(self.master.chat)
#--------------------------------------------------
class Jnl(customtkinter.CTkFrame):
    def __init__(self, master):
        customtkinter.CTkFrame.__init__(self, master)

        
         # create some widgets for page 1
    
        
        for widget in self.winfo_children():
            widget.destroy()
        def save_entry():
            entry_text = entry.get("1.0", customtkinter.END)
            now = datetime.now()
            date_time = now.strftime("%Y-%m-%d")
            filename = f"{date_time}.txt"
            with open(os.path.join("journal_entries", filename), "w") as f:
                f.write(entry_text)
            status_label.configure(text="Entry saved")
            refresh_calendar()

        def clear_entry():
            entry.delete("1.0", customtkinter.END)
            status_label.configure(text="Entry cleared")

        def refresh_calendar():
            all_entries = os.listdir("journal_entries")
            date_list = [filename.split(".")[0] for filename in all_entries]
            date_list.sort(reverse=True)
            if date_list:
                cal.set(date_list[0])
                cal.configure(values=date_list,justify='center')
            else:
                cal.set("")
                cal.configure(values=[],justify='center')

        def load_entry():
            selected_date = cal.get()
            filename = f"{selected_date}.txt"
            try:
                with open(os.path.join("journal_entries", filename), "r") as f:
                    entry_text = f.read()
                    entry.delete("1.0", customtkinter.END)
                    entry.insert("1.0", entry_text)
                    status_label.configure(text="Entry loaded", fg="blue")
            except FileNotFoundError:
                messagebox.showerror("Error", "No entry found for selected date.")

        def delete_entry():
            selected_date = cal.get()
            filename = f"{selected_date}.txt"
            try:
                os.remove(os.path.join("journal_entries", filename))
                entry.delete("1.0", customtkinter.END)
                status_label.configure(text="Entry deleted", fg="red")
                refresh_calendar()
            except FileNotFoundError:
                messagebox.showerror("Error", "No entry found for selected date.")

        def create_journal_entries_folder():
            if not os.path.exists("journal_entries"):
                os.mkdir("journal_entries")


        create_journal_entries_folder()

        entry_label = customtkinter.CTkLabel(self, text="Write your thoughts here:",height=50, width=100)
        entry_label.grid(row=0, column=0)
        

        entry = customtkinter.CTkTextbox(self, height=300, width=500, font=("Roboto", 12),corner_radius=10)
        entry.grid(row=1, column=0,ipady=50,ipadx=50)
        

        button_frame = customtkinter.CTkFrame(self, height=600, width=600)
        button_frame.grid(row=2, column=0,pady=5)
        

        save_button = customtkinter.CTkButton(button_frame, text="Save", command=save_entry,height=20, width=70, font=("Arial", 12))
        save_button.grid(row=5, column=0,pady=5)
        

        clear_button = customtkinter.CTkButton(button_frame, text="Clear", command=clear_entry,height=20, width=70, font=("Arial", 12))
        clear_button.grid(row=6, column=0,pady=5)
        

        status_label = customtkinter.CTkLabel(self, text="", font=("Arial", 12, "italic"), height=20, width=70)
        status_label.grid(row=7, column=0,pady=5)
        

        cal = customtkinter.CTkComboBox(self, width=150,height=30, font=("Arial", 12), state="readonly")
        cal.grid(row=8, column=0,pady=5)
        
        refresh_calendar()

        cal_button_frame = customtkinter.CTkFrame(self)
        cal_button_frame.grid(row=9, column=0,pady=5)
        

        load_button = customtkinter.CTkButton(cal_button_frame, text="Load Entry", command=load_entry,height=20, width=70, font=("Arial", 12))
        load_button.grid(row=10, column=0,pady=5)
        

        delete_button = customtkinter.CTkButton(cal_button_frame, text="Delete Entry",command=delete_entry, height=20, width=70,font=("Arial", 12))
        delete_button.grid(row=10, column=0,pady=5)
        
        self.button = customtkinter.CTkButton(self, text="Go to Main Page", command=self.go_to_main_page,height=20, width=70,font=("Arial", 12))
        self.button.grid(row=11, column=0, pady=5)
        
    def go_to_main_page(self):
    # navigate to the main page
        self.master.show_frame(self.master.main_page)
        

        
        
#--------------------------------------------------
class Chatter(customtkinter.CTkFrame):
    def __init__(self, master):
        customtkinter.CTkFrame.__init__(self, master)

         # create some widgets for page 2
        self.label = customtkinter.CTkLabel(self, text="Talk To Bhai")
        self.label.grid(row=0, column=0)
        
        def send():
            msg = EntryBox.get("1.0",'end-1c').strip()
            EntryBox.delete("0.0",END)

            if msg != '':
                ChatLog.configure(state=NORMAL)
                ChatLog.insert(END, "You: " + msg + '\n\n')
                got_intent = predict_intent(msg)
                if msg == 'goodbye':
                    ChatLog.insert(END, "BH.AI: " + get_response(msg,got_intent)[1] + '\n\n')
                    return
                res = get_response(msg,got_intent)[0]
                ChatLog.insert(END, "BH_AI: " + res + '\n\n')

                ChatLog.configure(state=DISABLED)
                ChatLog.yview(END)
        
        ChatLog = customtkinter.CTkTextbox(self, width=400,height=400 ,corner_radius=10)

        # ChatLog.config(state=DISABLED)
        scrollbar =  customtkinter.CTkScrollbar(self, command=ChatLog.yview(), cursor="heart")
        ChatLog['yscrollcommand'] = scrollbar.set

        EntryBox = customtkinter.CTkTextbox(self, width=300, corner_radius=5,height=20)

        SendButton = customtkinter.CTkButton(self, text="Send",command= send )

        scrollbar.grid(row=0, column=4,sticky='ns',pady=5)
        
        ChatLog.grid(row=0, column=0,pady=5,columnspan=3)
        
        EntryBox.grid(row=10, column=0,pady=5,padx=5)
        
        SendButton.grid(row=10, column=1,pady=5)
        

        self.button = customtkinter.CTkButton(self, text="Go to Main Page", command=self.go_to_main_page,width=15,font=("Arial", 12))
        self.button.grid(row=12, column=1, pady=5)
    def go_to_main_page(self):
    # navigate to the main page
        self.master.show_frame(self.master.main_page)

if __name__ == "__main__":
    app = Application()
    app.mainloop()
