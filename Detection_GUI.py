# -*- coding: utf-8 -*-
# @Time    : $2022-5-20
# @Author  : $Yidong Ding
# @File    : $Detection_GUI.py
# @Software: $Pycharm


import tkinter as tk
from feature_extraction import feature_extraction
from feature_extraction import is_http_url
from tkinter import messagebox
from classifiers import path
import joblib


class GUI:

    def __init__(self):
        self.root = tk.Tk()
        self.root.title('Phishing Website Detection')
        self.root.geometry("500x200+1100+150")
        # 运行的按钮
        self.Button0 = tk.Button(self.root, text="开始检测", command=self.event)
        # 退出的按钮
        self.Button1 = tk.Button(self.root, text="退出", command=self.root.destroy, bg="Gray")  # bg=颜色
        self.entry00 = tk.StringVar()  # 定义初始的字符串
        self.entry0 = tk.Entry(self.root, textvariable=self.entry00)
        self.Label0 = tk.Label(self.root, text="URL")

        self.interface()

    def interface(self):
        self.Button0.place(relx=0.2, x=200, y=20, relwidth=0.2, relheight=0.2)
        self.Button1.place(relx=0.2, x=100, y=100, relwidth=0.2, relheight=0.2)
        self.entry00.set("请输入URL")
        self.entry0.place(relx=0.2, x=8, y=20, relwidth=0.4, relheight=0.2)
        self.Label0.place(relx=0.05, x=10, y=30, relwidth=0.1, relheight=0.1)

    def event(self):
        """按钮事件,获取文本信息"""
        url = self.entry00.get()
        if is_http_url(url):
            fea = feature_extraction(url)
            clf = joblib.load(path)
            result = clf.predict(fea)
            if result == -1:
                messagebox.showinfo("Detection Result", "危险！钓鱼网址")
            else:
                messagebox.showinfo("Detection Result", "安全网址")
            pass
        else:
            messagebox.showinfo("Warning!", "输入不是URL，请重新输入")


if __name__ == '__main__':
    a = GUI()
    a.root.mainloop()






