# improt necessary packages
import tkinter as tk

class MailView:
    def __init__(self, title="Neural Style Transfer", bg="#FFFFFF"):
        self.window = tk.Tk()
        self.title = title
        self.bg = bg
        self.mail = "nobody"

    def create(self):
        self.window.wm_title(self.title)
        self.window.config(background=self.bg)
        self.window.attributes("-topmost", True)
        self.email = tk.Entry(width=40)
        self.email.focus_set()
        self.window.bind('<Return>', self.cmd_ok)
        self.lbl = tk.Label(master=self.window, text="Your email :")
        self.save = tk.Button(text="OK !", command=self.cmd_ok)
        self.lbl.pack()
        self.email.pack()
        self.save.pack()

    def cmd_ok(self, event=None):
        self.mail = self.email.get() if self.email.get() != "" else self.mail
        self.close()

    def show(self):
        self.window.geometry('250x60+800+400')
        self.window.mainloop()

    def close(self):
        self.window.quit()
        self.window.destroy()
