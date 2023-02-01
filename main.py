# -*- coding: cp1251 -*-
from PREDICTOR import Downloader, Predictor
import tkinter as tk
from tk import *
import numpy as np
#dictionaries for reactions
reaction_num = {"terrible" : [0, 0.22],
                "bad"      : [0.22, 0.35],
                "worse"    : [0.35, 0.5],
                "better"   : [0.5, 0.65],
                "good"     : [0.65, 0.9],
                "excellent": [0.9, 1]
                }
reaction_dict_eng = {"terrible" : ['That sounds awful...', 'How awful...', "I don't even want to read this...", 'It makes me sad...', "I don't want to comment on it..." ],
                     "bad"      : ['This is not good...', "I don't like this...", 'Bad...', 'It could be worse...'],
                     "worse"    : ['Well acceptable...', 'Not so bad...', 'Could be better...', "I'm not sure...", 'Not sure...'] ,
                     "better"   : ['Well acceptable...', 'Not so bad...', 'Could be better...', "I'm not sure...",'Not sure...', "I think it's good..."],
                     "good"     : ['Great!', "I think it's great!", "That's great!", "I'm happy about it!", 'Sounds great!'],
                     "excellent": ["Couldn't be better!", "I'm very happy about this!", "Awesome!", "Very good!"]
                     }
language = 'eng'
#color of prediction
def _from_rgb(rgb):
    r, g, b = rgb
    return f'#{r:02x}{g:02x}{b:02x}'
def pred_col(pred):
    color = _from_rgb((140, round(pred*150+50), 0))
    return color
#predict and react
def check():
    text = e.get()

    if text:
        pred = predictor.predict(text)
        if language == 'eng':
            if pred > reaction_num['terrible'][0] and pred < reaction_num['terrible'][1]:
                result= tk.Label(root, text= reaction_dict_eng["terrible"][np.random.randint(0,len(reaction_dict_eng["terrible"]))], font=('Arial', 18,''))
            if pred > reaction_num['bad'][0] and pred < reaction_num['bad'][1]:
                result= tk.Label(root, text= reaction_dict_eng["bad"][np.random.randint(0,len(reaction_dict_eng["bad"]))], font=('Arial', 18,''))
            if pred > reaction_num['worse'][0] and pred < reaction_num['worse'][1]:
                result= tk.Label(root, text= reaction_dict_eng["worse"][np.random.randint(0,len(reaction_dict_eng["worse"]))], font=('Arial', 18,''))
            if pred > reaction_num['better'][0] and pred < reaction_num['better'][1]:
                result= tk.Label(root, text= reaction_dict_eng["better"][np.random.randint(0,len(reaction_dict_eng["better"]))], font=('Arial', 18,''))
            if pred > reaction_num['good'][0] and pred < reaction_num['good'][1]:
                result= tk.Label(root, text= reaction_dict_eng["good"][np.random.randint(0,len(reaction_dict_eng["good"]))], font=('Arial', 18,''))
            if pred > reaction_num['excellent'][0] and pred < reaction_num['excellent'][1]:
                result= tk.Label(root, text= reaction_dict_eng["excellent"][np.random.randint(0,len(reaction_dict_eng["excellent"]))], font=('Arial', 18,''))

        result_float= tk.Label(root, text=str(round(pred, 3)), font=('Arial', 18,''), bg=pred_col(pred))
        canvas.create_window(350, 500, anchor=tk.CENTER, window=result,       width=600, height=60)
        canvas.create_window(350, 600, anchor=tk.CENTER, window=result_float, width=70,  height=35)
    else:
        def close_err():
            err.destroy()
        err = tk.Tk()
        err.title('Error')
        err.geometry('300x100')
        er_canvas = tk.Canvas(err,bg="white", width=300, height=100)
        er_canvas.pack(anchor=tk.CENTER, expand=1)

        close_btn = tk.Button(err, text = 'Ok', command = close_err)
        ErrorText = tk.Label( err, text="Error: You didn't paste anything.", font=('Arial', 10,''))

        er_canvas.create_window(270, 80, anchor=tk.CENTER, window=close_btn, width=30,  height=20)
        er_canvas.create_window(150, 40, anchor=tk.CENTER, window=ErrorText, width=250, height=50)

def delete_text():
    e.delete(0,'end')
def temp_text(event):
    e.delete(0,"end")
    
# create main window
root = tk.Tk()
root.iconbitmap("icon_tk.ico")
root.title('text checker')
root.geometry('700x800')
root.configure(background="#dcdcdc")

rootvar = tk.IntVar()
rootvar.set(0)
#create canvas in main window
canvas = tk.Canvas(bg="white", width=700, height=800)
canvas.pack(anchor=tk.CENTER, expand=1)

# add temporary text
e = tk.Entry(root, background='#c8c8c8')
e.insert(0, "Write / paste text here")
e.bind("<FocusIn>", temp_text)

# create widgets
ok_btn  = tk.Button(root, text = 'Ok',       command = check)
dlt_btn = tk.Button(root, text = 'Delete',   command = delete_text)
pst_btn = tk.Button(root, text = 'Paste',    command=lambda:e.event_generate("<<Paste>>"))
pst_txt = tk.Label( root, text = 'Paste text', font=('Arial', 18,''))
# add widgets to a window
canvas.create_window(350, 300, anchor=tk.CENTER, window=pst_btn,      width=160, height=60)
canvas.create_window(350, 150, anchor=tk.CENTER, window=pst_txt,      width=500, height=50)
canvas.create_window(350, 200, anchor=tk.CENTER, window=e,            width=500, height=50)
canvas.create_window(550, 300, anchor=tk.CENTER, window=ok_btn,       width=160, height=60)
canvas.create_window(150, 300, anchor=tk.CENTER, window=dlt_btn,      width=160, height=60)

# download LSTM
model_dir    = 'model/model.json' 
weghts_dir   = "model/model.h5"
tokenizer_dir= 'model/tokenizer.pickle'
downloader = Downloader()
predictor  = Predictor(
                        downloader.load_model(model_dir, weghts_dir),
                        downloader.load_tokenizer(tokenizer_dir)
                      )

# run the Tkinter event loop.
root.mainloop()
