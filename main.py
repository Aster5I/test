# -*- coding: utf-8 -*-

# import cv2
# import os
# import numpy as np
# import pickle
# import imutils
# from PIL import Image
# from kivy.app import App
# from kivy.uix.image import Image as KivyImage
# from kivy.graphics.texture import Texture
# from kivy.uix.button import Button
# from kivy.uix.boxlayout import BoxLayout
# from config import car_config as config
# from utility.preprocessing import ImageToArrayPreprocessor
# from utility.preprocessing import AspectAwarePreprocessor
# from utility.preprocessing import MeanPreprocessor
# from utility.utils import process_image as get_color
# import mxnet as mx
# from kivy.uix.label import Label
#
# # Загрузка модели и других ресурсов
# checkpointsPath = os.path.sep.join(["data/checkpoints", "vggnet"])
# model = mx.model.FeedForward.load(checkpointsPath, 65)
#
# le = pickle.loads(open(config.LABEL_ENCODER_PATH, "rb").read())
# sp = AspectAwarePreprocessor(width=224, height=224)
# mp = MeanPreprocessor(config.R_MEAN, config.G_MEAN, config.B_MEAN)
# iap = ImageToArrayPreprocessor(dataFormat="channels_first")
#
# #def predict_car_model(image_path, model, le, sp, mp, iap):
# #    image = cv2.imread(image_path)
# #   orig = image.copy()
# #    orig = imutils.resize(orig, width=min(500, orig.shape[1]))
# #    image = iap.preprocess(mp.preprocess(sp.preprocess(image)))
# #    image = np.expand_dims(image, axis=0)
# #
# #    preds = model.predict(image)[0]
# #    idxs = np.argsort(preds)[::-1][:5]
#
#  #   label = le.inverse_transform([idxs[0]])[0]
#   #  label = label.replace(":", " ")
#    # label = "{}: {:.2f}%".format(label, preds[idxs[0]] * 100)
# #    img_pil = Image.open(image_path)
# #    color = "color : " + get_color(img_pil)
# #    cv2.putText(orig, color, (10, 50), cv2.FONT_HERSHEY_SIMPLEX,
# #        0.6, (255, 0, 200), 2)
# #    cv2.putText(orig, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
# #        0.6, (0, 255, 0), 2)
#
#  #   return orig, label, color
#
# def predict_car_model(image_path, model, le, sp, mp, iap):
#     image = cv2.imread(image_path)
#     orig = image.copy()
#     orig = imutils.resize(orig, width=min(500, orig.shape[1]))
#     image = iap.preprocess(mp.preprocess(sp.preprocess(image)))
#     image = np.expand_dims(image, axis=0)
#
#     preds = model.predict(image)[0]
#     idxs = np.argsort(preds)[::-1][:3]
#
#     img_pil = Image.open(image_path)
#     color = "color: " + get_color(img_pil)
#
#     for i, idx in enumerate(idxs, start=1):
#         label = le.inverse_transform([idx])[0]
#         label = label.replace(":", " ")
#         label = "{}: {:.2f}%".format(label, preds[idx] * 100)
#         cv2.putText(orig, f"{i}. {label}", (10, 30 * i + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
#
#     # Изменение координаты Y для вывода цвета
#     cv2.putText(orig, color, (10, 30 * (len(idxs) + 1) + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 200), 2)
#
#     return orig, ', '.join(f"{le.inverse_transform([idx])[0]}: {preds[idx] * 100:.2f}%" for idx in idxs), color
#
#
# class CarPredictorApp(App):
#     def build(self):
#         layout = BoxLayout(orientation='vertical')
#         self.image_widget = KivyImage()
#         layout.add_widget(self.image_widget)
#         button = Button(text='Predict Car Model')
#         button.bind(on_press=self.predict)
#         layout.add_widget(button)
#         return layout
#
#     def predict(self, instance):
#         image_path = "test_img/125.jpg"  # Укажите путь к изображению
#         orig, label, color = predict_car_model(image_path, model, le, sp, mp, iap)
#         texture = self.cv2_to_kivy_texture(orig)
#         self.image_widget.texture = texture
#
#     def cv2_to_kivy_texture(self, orig):
#         orig = cv2.cvtColor(orig, cv2.COLOR_BGR2RGB)
#         buffer = cv2.flip(orig, 0).tobytes()
#         texture = Texture.create(size=(orig.shape[1], orig.shape[0]), colorfmt='rgb')
#         texture.blit_buffer(buffer, colorfmt='rgb', bufferfmt='ubyte')
#         return texture
#
# if __name__ == "__main__":
#     CarPredictorApp().run()

from kivy.uix.screenmanager import ScreenManager, Screen
from plyer import filechooser
import cv2
import os
import numpy as np
import pickle
import imutils
from PIL import Image
from kivy.uix.image import Image as KivyImage
from kivy.graphics.texture import Texture
from kivy.uix.button import Button
from config import car_config as config
from utility.preprocessing import ImageToArrayPreprocessor
from utility.preprocessing import AspectAwarePreprocessor
from utility.preprocessing import MeanPreprocessor
from utility.utils import process_image as get_color
import mxnet as mx
from kivymd.app import MDApp
from kivymd.uix.button import MDRaisedButton
from kivymd.uix.boxlayout import MDBoxLayout
from kivymd.uix.label import MDLabel
from kivy.core.window import Window
from kivy.utils import get_color_from_hex
from kivymd.uix.boxlayout import MDBoxLayout
from number_plate import recognize_license_plate
from kivy.core.text import LabelBase
LabelBase.register(name="CustomFont", fn_regular="data/Machine BT.ttf")
from kivy.uix.label import Label
from kivymd.uix.label import MDLabel


def predict_car_model(image_path, model, le, sp, mp, iap):
    image = cv2.imread(image_path)
    orig = image.copy()
    orig = imutils.resize(orig, width=min(500, orig.shape[1]))
    image = iap.preprocess(mp.preprocess(sp.preprocess(image)))
    image = np.expand_dims(image, axis=0)

    preds = model.predict(image)[0]
    idxs = np.argsort(preds)[::-1][:3]

    img_pil = Image.open(image_path)
    color = "Цвет: " + get_color(img_pil)

    for i, idx in enumerate(idxs, start=1):
        label = le.inverse_transform([idx])[0]
        label = label.replace(":", " ")
        label = "{}: {:.2f}%".format(label, preds[idx] * 100)
        cv2.putText(orig, f"{i}. {label}", (10, 30 * i + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Изменение координаты Y для вывода цвета
    cv2.putText(orig, color, (10, 30 * (len(idxs) + 1) + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 200), 2)

    #return orig, ', '.join(f"{le.inverse_transform([idx])[0]}: {preds[idx] * 100:.2f}%" for idx in idxs), color
    return orig, ', '.join(f"{le.inverse_transform([idx])[0]}: {preds[idx] * 100:.2f}%" for idx in idxs), color, idxs
# Создание основных экранов
class SelectImageScreen(Screen):
    def __init__(self, **kwargs):
        super(SelectImageScreen, self).__init__(**kwargs)
        layout = MDBoxLayout(orientation='vertical')
        self.image_widget = KivyImage()
        layout.add_widget(self.image_widget)
        self.select_image_button = MDRaisedButton(text='Выбрать изображение')
        self.select_image_button.bind(on_press=self.select_image)
        layout.add_widget(self.select_image_button)
        self.predict_button = MDRaisedButton(text='Продолжить', size_hint=(1, 0.1))
        self.predict_button.bind(on_press=self.go_to_predict_screen)
        layout.add_widget(self.predict_button)
        self.add_widget(layout)
    def select_image(self, instance):
        image_path = filechooser.open_file(title="Select Image")

        if image_path:
            self.manager.image_path = image_path[0]
            self.image_widget.source = self.manager.image_path
            self.select_image_button.text = "Выбрать другое изображение"

    def go_to_predict_screen(self, instance):
        if self.manager.image_path:
            self.manager.current = 'predict'
            self.manager.get_screen('predict').image_widget.source = self.manager.image_path
        else:
            print("No image selected")
class PredictScreen(Screen):
    def __init__(self, model, le, sp, mp, iap, **kwargs):
        super(PredictScreen, self).__init__(**kwargs)
        self.model = model
        self.le = le
        self.sp = sp
        self.mp = mp
        self.iap = iap
        layout = MDBoxLayout(orientation='vertical')
        self.image_widget = KivyImage()
        layout.add_widget(self.image_widget)
        self.predict_button = MDRaisedButton(text='Предсказать модель автомобиля')
        self.predict_button.bind(on_press=self.predict)
        layout.add_widget(self.predict_button)
        self.add_widget(layout)

    def predict(self, instance):
        # Распознавание номера
        recognized_license_plate = recognize_license_plate(self.manager.image_path)
        orig, label, color, idxs = predict_car_model(self.manager.image_path, self.model, self.le, self.sp, self.mp, self.iap)
        # Вывод распознанного номера на изображение
        cv2.putText(orig, recognized_license_plate, (10, 30 * (len(idxs) + 2) + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (255, 0, 0), 2)
        texture = self.cv2_to_kivy_texture(orig)
        self.image_widget.texture = texture
        self.manager.current = 'result'

    def cv2_to_kivy_texture(self, orig):
        orig = cv2.cvtColor(orig, cv2.COLOR_BGR2RGB)
        buffer = cv2.flip(orig, 0).tobytes()
        texture = Texture.create(size=(orig.shape[1], orig.shape[0]), colorfmt='rgb')
        texture.blit_buffer(buffer, colorfmt='rgb', bufferfmt='ubyte')
        return texture

class ResultScreen(Screen):
    def __init__(self, **kwargs):
        super(ResultScreen, self).__init__(**kwargs)
        layout = MDBoxLayout(orientation='vertical')
        self.image_widget = KivyImage()
        layout.add_widget(self.image_widget)
        # Создание виджета для вывода текста с результатами
        self.result_label = MDLabel(halign="center", theme_text_color="Secondary")
        self.result_label.font_style = "Body1"
        layout.add_widget(self.result_label)
        self.back_button = MDRaisedButton(text='Выбрать другое изображение')
        self.back_button.bind(on_press=self.back_to_select_image)
        layout.add_widget(self.back_button)
        self.add_widget(layout)

    def on_pre_enter(self, *args):
        self.image_widget.texture = self.manager.get_screen('predict').image_widget.texture
        # Заполнение текста с результатами
        # Заполнение текста с результатами
        predict_screen = self.manager.get_screen('predict')
        orig, label, color, idxs = predict_car_model(self.manager.image_path, predict_screen.model, predict_screen.le,
                                                     predict_screen.sp, predict_screen.mp, predict_screen.iap)
        recognized_license_plate = recognize_license_plate(self.manager.image_path)
        labels = label.split(', ')
        formatted_labels = '\n'.join(labels)
        self.result_label.text = f"{formatted_labels}\n{color}\n{recognized_license_plate}"
        self.result_label.texture_update()  # Обновление текстуры для корректного выравнивания

    def back_to_select_image(self, instance):
        self.manager.current = 'select'

class CarPredictorApp(MDApp):
    def build(self):
        Window.clearcolor = get_color_from_hex("#2C175C")
        # Загрузка модели и других ресурсов
        checkpointsPath = os.path.sep.join(["data/checkpoints", "vggnet"])
        model = mx.model.FeedForward.load(checkpointsPath, 65)
        le = pickle.loads(open(config.LABEL_ENCODER_PATH, "rb").read())
        sp = AspectAwarePreprocessor(width=224, height=224)
        mp = MeanPreprocessor(config.R_MEAN, config.G_MEAN, config.B_MEAN)
        iap = ImageToArrayPreprocessor(dataFormat="channels_first")

        sm = ScreenManager()
        sm.add_widget(SelectImageScreen(name='select'))
        sm.add_widget(PredictScreen(name='predict', model=model, le=le, sp=sp, mp=mp, iap=iap))
        sm.add_widget(ResultScreen(name='result'))
        return sm

if __name__ == "__main__":
    CarPredictorApp().run()