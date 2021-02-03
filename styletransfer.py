import io
import math
import torchvision.models as models
import random
import numpy as np
import os
import copy
import torch
import torchvision.transforms as transforms
import PIL
from PIL import Image
import math
import random
import numpy as np
import os
import copy
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
'''
Создадим модели, необходимые для обработки картинок (кроме GAN, она грузится отдельно в главном модуле bot.py)
'''
# напишем загрузчик
def image_loader(image_name, imsize):
    # приводим картинку к нужному нам размеру, переводим в тензор
    loader = transforms.Compose([
    transforms.Resize(imsize),  
    transforms.CenterCrop(imsize),
    transforms.ToTensor()])
    # загружаем
    image = Image.open(image_name)
    image = loader(image).unsqueeze(0)
    return image.to(device, torch.float)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# "Универсальная масочная функция". Работает быстрее написанных вручную масок, т.к. тензор маски не создается
# каждый раз с нуля
# путь приведен относительно репозитория 'Enotya_bot'. Возможно, Вам понадобится прописать свой путь к Masks, в зависимости от расположения данного каталога на
# Вашем устройстве. Будьте внимательны!
host = 'Masks/'

def masky(input, mask_type):
    _, f, h, _ = input.size()
    # загружаем файл с маской
    maskf = image_loader(host+mask_type+".bmp", h).cpu()
    mask = torch.tensor(np.array(maskf > 0.5, np.float32))
    # стакаем маску по числу фича-мапов у инпута          
    mask = mask.repeat(1, f, 1, 1)
    return mask.to(device)
   

# создаем зеркальную ей маску для применения второго стиля
def masky_dual(input, mask_type):
    _, f, h, _ = input.size()
    maskf = image_loader(host+mask_type+".bmp", h).cpu()
    mask = torch.tensor(np.array(maskf < 0.5, np.float32))          
    mask = mask.repeat(1, f, 1, 1)
    return mask.to(device)

# класс контент-лосса
class ContentLoss(nn.Module):

        def __init__(self, target,):
            super(ContentLoss, self).__init__()
            # это константа. Убираем ее из дерева вычислений
            self.target = target.detach()
            # инициализируем лосс
            self.loss = F.mse_loss(self.target, self.target )

        def forward(self, input):
            self.loss = F.mse_loss(input, self.target)
            return input

def gram_matrix(input):
        # Второй параметр -- число feature maps
        # (h,w) -- это размеры фича мапы (N=h*w)
        batch_size, f_map_num, h, w = input.size()  # batch size(=1)
        features = input.view(batch_size*f_map_num, h*w)  # преобразуем выход в вектор
        G = torch.mm(features, features.t())  # вычисляем ненормированную матрицу Грама

        # нормализуем элементы матрицы Грама, поделив на число элементов в каждой фича мапе
        return G.div(batch_size * h * w * f_map_num)

# класс для стилевых лоссов
class StyleLoss(nn.Module):
        def __init__(self, target_feature, style, option, mask_type = None):
            super(StyleLoss, self).__init__( )
            self.target = gram_matrix(target_feature).detach()
            self.style = style
            self.option = option
            self.mask_type = mask_type
            # инициализируем лосс
            self.loss = F.mse_loss(self.target, self.target)

        def forward(self, input):
            # прописываем применение выбранной маски и подсчет лосса 
            # для переноса разных стилей на разные части картинки 
            # (т.н. partial_style_transfer)
            if self.option == 'partial_style':
                if self.style == 1:
                    mask = masky(input, self.mask_type)
                elif self.style == 2:
                    mask = masky_dual(input, self.mask_type)     
                else:
                    raise RuntimeError('Incorrect style initialiation: {}'.format(self.style))     
                input1 = input.clone()
                input2 = input1 * mask
                G = gram_matrix(input2)
                self.loss = F.mse_loss(G, self.target)
            # прописываем подсчет лосса для одновременного переноса двух стилей 
            # на всю картинку 
            # (т.н. dual style transfer)
            elif self.option == 'dual_style':
                G = gram_matrix(input)
                self.loss = F.mse_loss(G, self.target) 
            else:
                raise RuntimeError('Unknown option for style: {}'.format(self.option))  
            return input

# вычисляем среднее и стандартное отклонение
cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(device)

# класс нормализации
class Normalization(nn.Module):
        def __init__(self, mean, std):
            super(Normalization, self).__init__()
            # .view среднее и стандартное отклонение к виду  [C x 1 x 1], чтобы
            # мы смогли вычислить их для стандартного тензора вида [B x C x H x W].
           
            self.mean = torch.tensor(mean).view(-1, 1, 1)
            self.std = torch.tensor(std).view(-1, 1, 1)

        def forward(self, img):
            # нормализуем картинку
            return (img - self.mean) / self.std

# загрузим предобученную vgg-модель
cnn = models.vgg19(pretrained=True).features[0:8].to(device).eval()

# создадим класс самой модели,осуществляющей перенос стиля в разных режимах: 
# один стиль (частный случай),  2 стиля одновременно, 2 стиля с маской
class MyStyleModel(nn.Module):
    def __init__(self, input_img, cnn, normalization_mean, normalization_std,
                      style1_img, style2_img, content_img, option, mask_type):
        super(MyStyleModel, self).__init__()
        self.input_img = input_img
        self.cnn = cnn
        self.normalization_mean = normalization_mean
        self.normalization_std = normalization_std
        self.style1_img = style1_img
        self.style2_img = style2_img
        self.content_img = content_img
        self.option = option
        self.mask_type = mask_type
        #self.style = 0

    # задаем функцию, возращающую модель переноса стиля и стилевые и контент-лоссы
    def get_style_model_and_losses(self):
        content_layers= ['conv_4']
        style_layers = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']
        global cnn
        cnn = copy.deepcopy(self.cnn)
        global style
        global model
        #global content_loss_
        content_losses = []
        style_losses1 = []
        style_losses2 = []
        # добавим в модель модуль нормализации
        normalization = Normalization(self.normalization_mean, self.normalization_std).to(device)
        model = nn.Sequential(normalization)
        i = 0  # переменная для подсчета сверточных слоев
        for layer in cnn.children():
            if isinstance(layer, nn.Conv2d):
                i += 1
                name = 'conv_{}'.format(i)
            elif isinstance(layer, nn.ReLU):
                name = 'relu_{}'.format(i)
                # Переопределим Relu уровень 
                # для корректной работы со стилевыми слоями
                layer = nn.ReLU(inplace=False)
            elif isinstance(layer, nn.MaxPool2d):
                name = 'pool_{}'.format(i)
            elif isinstance(layer, nn.BatchNorm2d):
                name = 'bn_{}'.format(i)
            else:
                raise RuntimeError('Unrecognized layer: {}'.format(layer.__class__.__name__))

            model.add_module(name, layer)

            if name in content_layers:
                # добавляем в модель слой  content loss-а:
                target = model(self.content_img).detach()
                сontent_loss = ContentLoss(target)
                model.add_module("content_loss_{}".format(i), сontent_loss)
                content_losses.append(сontent_loss)
            # добавляем в модель слои style loss-ов:
            if name in style_layers:
                # если мы осуществляем перенос разных стилей на разные части картинки,
                # то введем переменную style, которая будет определять, с какой именно
                # частью картинки мы работаем -- первый стилевой слой используется 
                # для одной части изображения, а второй стилевой слой -- для другой.
                # Соответствующие части картинки выделяются маской
                # (так что здесь мы еще указываем и тип маски)
                if self.option == 'partial_style':
                # добавили первый слой 
                    style = 1 # режим первого стиля
                    target_feature1 = model(self.style1_img).detach()
                    style_loss1 = StyleLoss(target_feature1, style, self.option, self.mask_type)
                    model.add_module("style_loss1_{}".format(i), style_loss1)
                    style_losses1.append(style_loss1)
                    # добавили второй слой
                    style = 2 # режим второго стиля
                    target_feature2 = model(self.style2_img).detach()
                    style_loss2 = StyleLoss(target_feature2, style, self.option, self.mask_type)
                    model.add_module("style_loss2_{}".format(i), style_loss2)
                    style_losses2.append(style_loss2)
                    # если же мы осуществляем одновременный перенос двух стилей на всю картинку,
                    # просто создаем 2 стилевых слоя и последовательно  прогоняем  
                    # наше изображение целиком через оба слоя.
                    # маска нам здесь больше не нужна, и мы ее даже не инициализируем.
                    # Переменная style тоже не нужна, и мы ее зануляем
                elif self.option == 'dual_style':
                    style = 0 # одновременный стилевой режим
                    # добавили первый слой
                    target_feature1 = model(self.style1_img).detach()
                    style_loss1 = StyleLoss(target_feature1, style, self.option)
                    model.add_module("style_loss1_{}".format(i), style_loss1)
                    style_losses1.append(style_loss1)
                    # добавили второй слой
                    target_feature2 = model(self.style2_img).detach()
                    style_loss2 = StyleLoss(target_feature2, style, self.option)
                    model.add_module("style_loss2_{}".format(i), style_loss2)
                    style_losses2.append(style_loss2)
                else:
                    raise RuntimeError('Unrecognized style: {}'.format(self.option))

        #выбрасываем все уровни VGG-ки после последенего style loss или content loss
        for i in range(len(model) - 1, -1, -1):
            if isinstance(model[i], ContentLoss) or isinstance(model[i], StyleLoss):
                break

        model = model[:(i + 1)]

        return model, style_losses1, style_losses2, content_losses
    
    # задаем оптимизатор
    def get_input_optimizer(self):
       
        #добоваляет содержимое тензора катринки в список изменяемых оптимизатором параметров
        optimizer = optim.LBFGS([self.input_img.requires_grad_()]) 
        return optimizer

    # задаем функцию обучения модели
    def run_style_transfer(self, num_steps=750,
                        style_weight1=100000, style_weight2=100000, content_weight=1):
        print('Building the style transfer model..')
        model, style_losses1, style_losses2, content_losses = self.get_style_model_and_losses()
        optimizer = self.get_input_optimizer()
        eps = 1e-9
        print('Optimizing..')
        run = [0]
        while run[0] <= num_steps:
        # Это функция, которая вызывается во время каждого прохода, чтобы пересчитать loss.
        # Без нее ничего не получется, так как у нас своя -- не библиотечная -- функция ошибки
            def closure():
                # это для того, чтобы значения тензора картинки не выходили за пределы [0;1]
                self.input_img.data.clamp_(eps, 1-eps)
                # обнуляем градиент
                optimizer.zero_grad()

                # прогоняем картинку через модель
                model(self.input_img)

                # считаем лоссы
                style_score1 = 0
                style_score2 = 0
                content_score = 0

                for sl in style_losses1:
                    style_score1 += sl.loss
                
                for sl in style_losses2:
                    style_score2 += sl.loss

                for cl in content_losses:
                    content_score += cl.loss
                
                # взвешивание ошибки в зависимости от типа обработки картинки
                if self.option == 'partial_style':
                    style_score1 *= style_weight1
                    style_score2 *= style_weight1
                    content_score *= content_weight
                elif self.option == 'dual_style':
                    style_score1 *= style_weight1
                    style_score2 *= 0.7 * style_weight2
                    content_score *= content_weight
                else:
                    raise RuntimeError('Unrecognized style: {}'.format(self.option))
                # строим лосс
                loss = style_score1 + style_score2 + content_score
                # и вычисляем по нему градиент
                loss.backward()

                run[0] += 1
                if run[0] % 50 == 0:
                    print("run {}:".format(run))
                    print('Style Loss1 : {:4f} Style Loss2 : {:4f} Content Loss: {:4f}'.format(
                        style_score1.item(), style_score2.item(), content_score.item()))
                    print()
                

                return style_score1 + style_score2 + content_score
            # шаг градиентного спуска
            optimizer.step(closure)

        # еще раз корректируем картинку
        self.input_img.data.clamp_(eps, 1-eps)
        return self.input_img
