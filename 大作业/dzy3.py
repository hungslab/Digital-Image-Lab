import tkinter as tk
from tkinter import Button, Canvas, filedialog
import cv2
import numpy as np
from PIL import Image, ImageTk

class WinGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.__win()
        self.tk_button_load_image = self.__tk_button_load_image(self)
        self.tk_button_gray_image = self.__tk_button_gray_image(self)
        self.tk_button_fourier_transform = self.__tk_button_fourier_transform(self)
        self.tk_button_linear_transform = self.__tk_button_linear_transform(self)
        self.tk_button_nonlinear_transform = self.__tk_button_nonlinear_transform(self)
        self.tk_canvas_original = self.__tk_canvas_original(self)
        self.tk_canvas_transformed = self.__tk_canvas_transformed(self)
        self.label_a = self.__label_a(self)
        self.label_b = self.__label_b(self)
        self.entry_a = self.__entry_a(self)
        self.entry_b = self.__entry_b(self)
        self.label_c = self.__label_c(self)
        self.label_gamma = self.__label_gamma(self)
        self.entry_c = self.__entry_c(self)
        self.entry_gamma = self.__entry_gamma(self)
        self.image = None

    def __win(self):
        self.title("图像处理")
        width = 600
        height = 500
        screenwidth = self.winfo_screenwidth()
        screenheight = self.winfo_screenheight()
        geometry = '%dx%d+%d+%d' % (width, height, (screenwidth - width) / 2, (screenheight - height) / 2)
        self.geometry(geometry)
        self.resizable(width=False, height=False)

    def __label_a(self, parent):
        label = tk.Label(parent, text="参数 a:")
        label.place(x=280, y=225)
        return label

    def __label_b(self, parent):
        label = tk.Label(parent, text="参数 b:")
        label.place(x=280, y=255)
        return label

    def __entry_a(self, parent):
        entry = tk.Entry(parent, width=10)
        entry.place(x=340, y=225)
        return entry

    def __entry_b(self, parent):
        entry = tk.Entry(parent, width=10)
        entry.place(x=340, y=255)
        return entry

    def __label_c(self, parent):
        label = tk.Label(parent, text="参数 c:")
        label.place(x=280, y=285)
        return label

    def __label_gamma(self, parent):
        label = tk.Label(parent, text="参数 γ:")
        label.place(x=280, y=315)
        return label

    def __entry_c(self, parent):
        entry = tk.Entry(parent, width=10)
        entry.place(x=340, y=285)
        return entry

    def __entry_gamma(self, parent):
        entry = tk.Entry(parent, width=10)
        entry.place(x=340, y=315)
        return entry

    def __tk_button_load_image(self, parent):
        btn = Button(parent, text="载入图片", takefocus=False, command=self.load_image)
        btn.place(x=439, y=63, width=120, height=55)
        return btn

    def __tk_button_gray_image(self, parent):
        btn = Button(parent, text="灰度图像输出", takefocus=False, command=self.convert_to_gray)
        btn.place(x=439, y=141, width=120, height=55)
        return btn

    def __tk_button_fourier_transform(self, parent):
        btn = Button(parent, text="傅里叶变换图像输出", takefocus=False, command=self.apply_fourier_transform)
        btn.place(x=439, y=383, width=120, height=55)
        return btn

    def __tk_button_linear_transform(self, parent):
        btn = Button(parent, text="线性灰度变换", takefocus=False, command=self.linear_transform)
        btn.place(x=440, y=220, width=120, height=55)
        return btn

    def __tk_button_nonlinear_transform(self, parent):
        btn = Button(parent, text="非线性灰度变换", takefocus=False, command=self.nonlinear_transform)
        btn.place(x=438, y=301, width=120, height=55)
        return btn

    def __tk_canvas_original(self, parent):
        canvas = Canvas(parent, bg="#aaa")
        canvas.place(x=60, y=38, width=200, height=200)
        return canvas

    def __tk_canvas_transformed(self, parent):
        canvas = Canvas(parent, bg="#aaa")
        canvas.place(x=59, y=257, width=200, height=200)
        return canvas

    def load_image(self):
        file_path = filedialog.askopenfilename()
        if file_path:
            self.image = cv2.imread(file_path)
            self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
            self.display_image(self.image, self.tk_canvas_original)

    def convert_to_gray(self):
        if self.image is not None:
            gray_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
            self.display_image(gray_image, self.tk_canvas_transformed)

    def apply_fourier_transform(self):
        if self.image is not None:
            gray_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
            # 快速傅里叶变换算法得到频率分布
            f_transform = np.fft.fft2(gray_image)
            # 默认结果中心点位置是在左上角, 调用fftshift()函数转移到中间位置
            f_shift = np.fft.fftshift(f_transform)
            # fft结果是复数, 其绝对值结果是振幅
            magnitude_spectrum = 20 * np.log(np.abs(f_shift))
            self.display_image(magnitude_spectrum, self.tk_canvas_transformed)

    def linear_transform(self):
        if self.image is not None:
            gray_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
            a = float(self.entry_a.get()) if self.entry_a.get() else 2.0
            b = float(self.entry_b.get()) if self.entry_b.get() else 0
            normalized_image = np.float64(gray_image)
            linear_stretch = a * normalized_image + b
            self.display_image(linear_stretch, self.tk_canvas_transformed)

    def nonlinear_transform(self):
        if self.image is not None:
            gray_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
            c = float(self.entry_c.get()) if self.entry_c.get() else 1
            gamma = float(self.entry_gamma.get()) if self.entry_gamma.get() else 0.5
            normalized_image = np.float64(gray_image)
            power_law = c * np.power(normalized_image, gamma)
            self.display_image(power_law, self.tk_canvas_transformed)

    def display_image(self, image, canvas):
        image = Image.fromarray(image)
        photo = ImageTk.PhotoImage(image=image)
        canvas.create_image(0, 0, anchor=tk.NW, image=photo)
        canvas.image = photo


if __name__ == "__main__":
    win = WinGUI()
    win.mainloop()
