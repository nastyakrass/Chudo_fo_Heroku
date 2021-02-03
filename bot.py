# -*- coding: utf-8 -*-
# создаем бота
# TOKEN = 'YOUR_TOKEN'
TOKEN = '1492842274:AAFIUViv4tTuRBS3hPJ_Qq4zdmS-vHxJUZ4'
import telebot;
bot = telebot.TeleBot(TOKEN);

from telebot.util import async_dec
import io
'''
Подгрузим модели, необходимые для обработки картинок
'''
imsize = 256
from styletransfer import *
# загружаем GAN для суперрезолюции
'''
На Heroku gan не влезает!!
'''
#from ISR.models import RDN
#gan = RDN(weights='psnr-small')

#from ISR.models import RRDN
#gan = RRDN(weights='gans')


# путь приведен относительно репозитория 'Enotya_bot'. Возможно, Вам понадобится прописать свой путь к Masks, в зависимости от расположения данного каталога на
# Вашем устройстве. Будьте внимательны!
# host = 'Masks/' -- данная переменная уже определена в styletransfer.py

''' 
Теперь строим асинхронного бота с бибилиотекой Телебот!
Здесь нам поможет декоратор @async_dec()
Конструкция тестировалась -- она работает асинхронно :)
'''
# функция, которая выводит на экран пользователя меню бота после каждой отработки
@async_dec()
def signal(message):
    bot.send_message(message.from_user.id, "Что теперь будем полоскать? Тазик готов!")
    keyboard = types.InlineKeyboardMarkup()
    # Делаем кнопки для различных режимов переноса стиля
    key_1st= types.InlineKeyboardButton(text='Перенесем 1 стиль', callback_data='1st')
    # И добавляем кнопку на экран
    keyboard.add(key_1st)
    key_2st= types.InlineKeyboardButton(text='Перенесем 2 стиля с маской', callback_data='2st')
    # И добавляем кнопку на экран
    keyboard.add(key_2st)
    key_doublest= types.InlineKeyboardButton(text='Перенесем одновременно 2 стиля на картинку', callback_data='double')
    # И добавляем кнопку на экран
    keyboard.add(key_doublest)
    key_gan = types.InlineKeyboardButton(text='Применим GAN', callback_data='gan')
    # И добавляем кнопку на экран
    keyboard.add(key_gan)
    # Выводим клавиатуру
    bot.send_message(message.from_user.id, text='Выбери действие, порадуй Енотика', reply_markup=keyboard)

# функция для команды "help"
@async_dec()
def signal_h(message):
    bot.send_message(message.from_user.id, "Не стесняйся -- выбирай любую понравившуюся опцию из списка и далее следуй инструкциям от Еноти. Для загрузки фоток можно пользоваться и Forward'ом. С Любовью, твой Енот")
    keyboard = types.InlineKeyboardMarkup()
    # Клавиатура
    key_1st= types.InlineKeyboardButton(text='Перенесем 1 стиль', callback_data='1st')
    keyboard.add(key_1st)
    key_2st= types.InlineKeyboardButton(text='Перенесем 2 стиля с маской', callback_data='2st')
    keyboard.add(key_2st)
    key_doublest= types.InlineKeyboardButton(text='Перенесем одновременно 2 стиля на картинку', callback_data='double')
    keyboard.add(key_doublest)
    key_gan = types.InlineKeyboardButton(text='Применим GAN: Super Resolution', callback_data='gan')
    keyboard.add(key_gan)
    bot.send_message(message.from_user.id, text='Выбери действие, порадуй Енотика', reply_markup=keyboard)

import io
# функция выбора размера картинок (рекомендован размер 256 на 256)
@async_dec()
def size_im(message):
    bot.send_message(message.from_user.id, "Выбери для начала размер выходных картинок. Ты всегда можешь его изменить, вновь нажав на команду 'start'. С Любовью, твой Енот")
    keyboard = types.InlineKeyboardMarkup()
    key_1st= types.InlineKeyboardButton(text='256 на 256 (рекомендовано!)', callback_data='large')
    keyboard.add(key_1st)
    key_2st= types.InlineKeyboardButton(text='128 на 128', callback_data='small')
    keyboard.add(key_2st)
    bot.send_message(message.from_user.id, text='Выбери свой размерчик! (по умолчанию -- 256)', reply_markup=keyboard)

# функция перевода тензора в PIL-картинку: нужна для отправки обрабтанного изображения пользователю
def get_res(output):
    bio = io.BytesIO()
    bio.name = 'output_m.jpeg'
    unloader = transforms.ToPILImage()
    image = output.clone()   
    image = image.squeeze(0) 
    image = unloader(image)
    image.save(bio, 'jpeg')
    bio.seek(0)
    return image

# напишем функцию, которая по запросу пользователя выводит 3 ссылочки с ответами на его вопрос
from googlesearch import search
@async_dec()
def find_me(message, query):
    # результаты Google Search 
    search_result_list = list(search(query, tld="co.in", num=10, stop=3, pause=1))
    if len(search_result_list) == 0:
        bot.send_message(message.chat.id, "Енотик этого не знает :( Попробуй спросить что-нибудь еще или переформулируй запрос" )
    else:
        bot.send_message(message.chat.id, "Держи!")
        bot.send_message(message.chat.id, search_result_list[0])
        bot.send_message(message.chat.id, search_result_list[1])
        bot.send_message(message.chat.id, search_result_list[2])
        
# создаем текстовые файлики для записи и считывания текущего режима обработки и т.п.
with open(host+"data.txt", "w") as f:
    f.write('BotEnot\n')
with open(host+"size.txt", "w") as f:
    f.write('256')
with open(host+"query.txt", "w") as f:
    f.write('Где живет енот')
# обработка команды "start"
@async_dec()
@bot.message_handler(commands=['start'])
def start_message(message):
    bot.send_message(message.chat.id, 'Привет, я -- Енотя, ученый Енот-полоскун, который умеет полоскать фотки разными интересными способами. К тому же, я могу делать Super Resolution картинки без потери качества. Напиши мне что-нибудь, и я порадую тебя, ведь Еноты -- они такие'+'\U0001F49B')
    # шлём привественную картинку
    with open(host+"poloskun.jpg", "rb") as file:
        data = file.read()
    bot.send_photo(message.chat.id, photo=data)
    size_im(message)

# обработка команды "help"
@async_dec()
@bot.message_handler(commands=['help'])
def start_message(message):
    bot.send_message(message.chat.id, 'Привет, ученый Енот-полоскун бежит на помощь :)')
    signal_h(message)

# обработка команды "request"
@async_dec()
@bot.message_handler(commands=['request'])
def knowledge(message):
    bot.send_message(message.chat.id, "Спроси о чем-нибудь Енотю, и он пришлет полезные ссылочки об этом!")
    with open(host+"query.txt", "w") as f:
        f.write('query')


from telebot import types
# текстовый обработчик, вывод меню
@async_dec()
@bot.message_handler(content_types=['text'])
def get_text_messages(message):
    with open(host+'query.txt', 'r') as file:
        file_content=file.readlines()
        for line in file_content:
            ask = line.strip()
    if ask == "query":
        find_me(message, message.text)
        with open(host+"query.txt", "w") as f:
            f.write('Где живет енот')
    else:
        bot.send_message(message.from_user.id, "Сейчас я отполоскаю твою фотку так, как ты захочешь")
        bot.send_message(message.from_user.id, "Я приготовил тазик :-)")
        keyboard = types.InlineKeyboardMarkup()
        key_1st= types.InlineKeyboardButton(text='Перенесем 1 стиль', callback_data='1st')
        keyboard.add(key_1st)
        key_2st= types.InlineKeyboardButton(text='Перенесем 2 стиля с маской', callback_data='2st')
        keyboard.add(key_2st)
        key_doublest= types.InlineKeyboardButton(text='Перенесем одновременно 2 стиля на картинку', callback_data='double')
        keyboard.add(key_doublest)
        key_gan = types.InlineKeyboardButton(text='Применим GAN', callback_data='gan')
        keyboard.add(key_gan)
        bot.send_message(message.from_user.id, text='Выбери действие, порадуй Енотика', reply_markup=keyboard)


# обработка нажатий на виртуальные кнопки выбора режимов обработки, размера картинок, вида масок и т.п.
@async_dec()
@bot.callback_query_handler(func=lambda call: True)
def callback_worker(call):
    if call.data == "large":
        bot.send_message(call.message.chat.id, "Вызов принят!")
        bot.send_message(call.message.chat.id, "Теперь напиши мне что-нибудь хорошее, а то я не буду полоскать! Или жми на косую черту внизу строки ввода и выбирай команду 'Помощь Полоскуна'")
        with open(host+"size.txt", "w") as f:
            f.write('256')
        bot.answer_callback_query(call.id, text  = " ") 
    if call.data == "small":
        bot.send_message(call.message.chat.id, "Ух, минималист!")
        bot.send_message(call.message.chat.id, "Теперь напиши мне что-нибудь хорошее, а то я не буду полоскать! Или жми на косую черту внизу строки ввода и выбирай команду 'Помощь Полоскуна'")
        with open(host+"size.txt", "w") as f:
            f.write('128') 
        bot.answer_callback_query(call.id, text  = " ")
    if call.data == "gan":
        bot.send_message(call.message.chat.id, "Загрузи свою фотку на обработку!")
        with open(host+"data.txt", "w") as f:
            f.write('Gan\n') 
        bot.answer_callback_query(call.id, text  = " ") 
    if call.data == "double":
        bot.send_message(call.message.chat.id, "Загрузи свою фотку на обработку!")
        with open(host+"data.txt", "w") as f:
            f.write('Double\n') 
        bot.answer_callback_query(call.id, text  = " ")   
    elif call.data == "1st":
        bot.send_message(call.message.chat.id, "Загрузи свою фотку на обработку!")
        with open(host+"data.txt", "w") as f:
            f.write('1st\n')   
        bot.answer_callback_query(call.id, text  = " ")
    elif call.data == "2st":
        with open(host+"data.txt", "w") as f:
            f.write('2st\n') 
        keyboard = types.InlineKeyboardMarkup()
        # клавиатура для выбора типов масок: доступно 4 опции
        key_1st= types.InlineKeyboardButton(text='Диагональная маска', callback_data='diag')
        keyboard.add(key_1st)
        key_2st= types.InlineKeyboardButton(text='Интересная маска', callback_data='interest')
        keyboard.add(key_2st)
        key_doublest= types.InlineKeyboardButton(text='Вертикальная маска', callback_data='vert')
        keyboard.add(key_doublest)
        key_gan = types.InlineKeyboardButton(text='Горизонтальная маска', callback_data='hor')
        keyboard.add(key_gan)
        bot.send_message(call.message.chat.id, text='Выбери масочку, не серди Енотика', reply_markup=keyboard)  
        bot.answer_callback_query(call.id, text  = " ")
        
    elif call.data == "diag":
        with open(host+"mask.txt", "w") as f:
            f.write('diag\n') 
        bot.send_message(call.message.chat.id, "А вот теперь загрузи свою фотку на обработку!")
        bot.answer_callback_query(call.id, text  = " ")
    elif call.data == "interest":
        with open(host+"mask.txt", "w") as f:
            f.write('interesting\n') 
        bot.send_message(call.message.chat.id, "А вот теперь загрузи свою фотку на обработку!")
        bot.answer_callback_query(call.id, text  = " ")
    elif call.data == "hor":
        with open(host+"mask.txt", "w") as f:
            f.write('hor\n') 
        bot.send_message(call.message.chat.id, "А вот теперь загрузи свою фотку на обработку!")
        bot.answer_callback_query(call.id, text  = " ")
    elif call.data == "vert":
        with open(host+"mask.txt", "w") as f:
            f.write('vert\n') 
        bot.send_message(call.message.chat.id, "А вот теперь загрузи свою фотку на обработку!")
        bot.answer_callback_query(call.id, text  = " ")

# обработка неподдерживаемого контента
@async_dec()
@bot.message_handler(content_types=['audio', 'document'])        
def content(message):
    bot.send_message(message.from_user.id, "Енотя такое не полоскает! Следуй моим инструкциям или нажми на косую черту в правом углу строки ввода и выбери команду 'Помощь Полоскуна' ")    

# обработка стикеров
@async_dec()
@bot.message_handler(content_types=['sticker'])
def get_reply(message):
    signal_h(message)
    
# переключатели режимов обработки фото
s = 0
v = 0
l = 0
m = 0

# обработчик фото
@async_dec()
@bot.message_handler(content_types=['photo'])        
def photo(message):
    global fileID
    global fileID1
    global fileID2
    global content_img
    global style1_img
    global s
    global v
    global m
    global l
    global mask
    global imsize
    # считываем текущий размер картинки
    with open(host+'size.txt', 'r') as file:
        file_content=file.readlines()
        for line in file_content:
            imsize = int(line.strip())
    # считываем текущий режим обработки
    with open(host+'data.txt', 'r') as file:
        file_content=file.readlines()
        for line in file_content:
            vals = line.strip()
    # Если выбран режим "Gan : SuperResolution"
    if vals == 'Gan':
        # загружаем фото на обработку
        if (v == 0) :
            bot.send_message(message.from_user.id, "Жду фотку на обработку!")
            print('Superresolution')
            fileID = message.photo[-1].file_id
            file_info = bot.get_file(fileID)
            downloaded_file = bot.download_file(file_info.file_path)
            img = Image.open(io.BytesIO(downloaded_file))
            v += 1
            bot.send_message(message.from_user.id, "Енотя доволен!")
            print(v)
        # обрабатываем
        if v == 1:
            bot.send_message(message.from_user.id, "Енотя полоскает...")
            v = 0
            # выводим результат обычной бикубической интерполяции -- понижает размерность
            img = img.resize((img.size[0]-120, img.size[1]-120), Image.BICUBIC)
            bio = io.BytesIO()
            bio.name = 'output.jpeg'
            img.save(bio, 'jpeg')
            bio.seek(0)
            bot.send_message(message.from_user.id, "Посмотри  на результат обычной бикубической интерполяции (я сжал фото для более наглядной демонстрации)")
            bot.send_photo(message.from_user.id, photo=img)
            # а теперь применяем GAN  и выводим результат
            u = img.resize((img.size[0]+150, img.size[1]+150), Image.BICUBIC)
            bio = io.BytesIO()
            bio.name = 'output_f.jpeg'
            u.save(bio, 'jpeg')
            bio.seek(0)
            bot.send_message(message.from_user.id, "А теперь взгляни на работу GAN")
            bot.send_photo(message.from_user.id, photo=u)
            # радуемся :)
            bot.send_message(message.from_user.id, "Видишь, Енотя не врет, качество Super resolution действительно классное!")
            bot.send_message(message.from_user.id, "За пушистый хвост енотий поднимем бокальчик!" + "\U0001F942")
            # выводим меню
            signal(message)
    # Если выбран режим "Перенесем 2 стиля с маской"
    if vals == '2st':
        print(s)
        print('With mask')
        # загружаем фото на обработку
        if (s == 0) :
            bot.send_message(message.from_user.id, "Жду фотку на обработку!")
            with open(host+'mask.txt', 'r') as file:
                file_content=file.readlines()
                for line in file_content:
                    mask = line.strip()
            fileID = message.photo[-1].file_id
            file_info = bot.get_file(fileID)
            downloaded_file = bot.download_file(file_info.file_path)
            img = io.BytesIO(downloaded_file)
            content_img = image_loader(img, imsize)
            s += 1
            bot.send_message(message.from_user.id, "Енотя доволен!")
        # загружаем первую стилевую фотку   
        if (s == 1):
            fileID1 = message.photo[-1].file_id
            if (fileID != fileID1):
                bot.send_message(message.from_user.id, "Енотя очень доволен!")   
                file_info = bot.get_file(fileID1)
                downloaded_file = bot.download_file(file_info.file_path)
                img1 = io.BytesIO(downloaded_file)
                style1_img = image_loader(img1, imsize)
                s += 1
        if s == 1:
            bot.send_message(message.from_user.id, "Теперь загрузи первую стилевую фотку на обработку!") 
        # загружаем вторую стилевую фотку 
        if (s == 2):
            fileID2 = message.photo[-1].file_id    
            if (fileID != fileID1) and  (fileID != fileID2) and  (fileID1 != fileID2):
                bot.send_message(message.from_user.id, "Енотя чрезвычайно доволен!")   
                file_info = bot.get_file(fileID2)
                downloaded_file = bot.download_file(file_info.file_path)
                img2 = io.BytesIO(downloaded_file)
                style2_img = image_loader(img2, imsize)
                s += 1
        if 2 <= s < 3:
            bot.send_message(message.from_user.id, "Теперь загрузи вторую стилевую фотку на обработку!") 
        # обрабатываем
        if s == 3:
            bot.send_message(message.from_user.id, "Енотя полоскает...")
            s = 0
            input_img = content_img.clone()
            # создаем экземпляр нашего класса для обработки с маской и переносим стиль
            st_transf_model = MyStyleModel(input_img, cnn, cnn_normalization_mean, cnn_normalization_std,
                      style1_img, style2_img, content_img, option ='partial_style', mask_type = mask)
            output = st_transf_model.run_style_transfer(num_steps=100)
            # отправляем результат и радуемся)
            image = get_res(output)
            bot.send_photo(message.from_user.id, photo=image)
            bot.send_message(message.from_user.id, "За пушистый хвост енотий поднимем бокальчик!"+"\U0001F942")
            # выводим меню
            signal(message)
    # Если выбран режим "Перенесем 2 стиля одновременно"
    elif vals == 'Double':
        print(m)
        print('Double')
        # загружаем фото на обработку
        if (m == 0) :
            bot.send_message(message.from_user.id, "Жду фотку на обработку!")
            fileID = message.photo[-1].file_id
            file_info = bot.get_file(fileID)
            
            downloaded_file = bot.download_file(file_info.file_path)
            img = io.BytesIO(downloaded_file)
            content_img = image_loader(img, imsize)
            m += 1
            bot.send_message(message.from_user.id, "Енотя доволен!")
        # загружаем первую стилевую фотку   
        if (m == 1):
            fileID1 = message.photo[-1].file_id
            if (fileID != fileID1):
                bot.send_message(message.from_user.id, "Енотя очень доволен!")   
                file_info = bot.get_file(fileID1)
                downloaded_file = bot.download_file(file_info.file_path)
                img1 = io.BytesIO(downloaded_file)
                style1_img = image_loader(img1, imsize)
                m += 1
        
        if (m == 1):
            bot.send_message(message.from_user.id, "Теперь загрузи первую стилевую фотку на обработку!") 
        # загружаем вторую стилевую фотку 
        if (m == 2):
            fileID2 = message.photo[-1].file_id    
            if (fileID != fileID1) and  (fileID != fileID2) and  (fileID1 != fileID2):
                bot.send_message(message.from_user.id, "Енотя чрезвычайно доволен!")   
                file_info = bot.get_file(fileID2)
                downloaded_file = bot.download_file(file_info.file_path)
                img2 = io.BytesIO(downloaded_file)
                style2_img = image_loader(img2, imsize)
                m += 1
        if 2 <= m < 3:
            bot.send_message(message.from_user.id, "Теперь загрузи вторую стилевую фотку на обработку!") 
        # обрабатываем
        if m == 3:
            bot.send_message(message.from_user.id, "Енотя полоскает...")
            m = 0
            input_img = content_img.clone()
            # создаем экземпляр нашего класса для обработки с двумя стилями без маски  -- и переносим стиль
            st_transf_model = MyStyleModel(input_img, cnn, cnn_normalization_mean, cnn_normalization_std,
                      style1_img, style2_img, content_img, option='dual_style', mask_type=None)
            output = st_transf_model.run_style_transfer(num_steps=100)
            # отправляем результат и радуемся)
            image = get_res(output)
            bot.send_photo(message.from_user.id, photo=image)
            bot.send_message(message.from_user.id, "За пушистый хвост енотий поднимем бокальчик!"+"\U0001F942")
            # выводим меню
            signal(message)
    # Если выбран режим "Перенесем 1 стиль"
    elif vals == '1st':
        print(l)
        print('1 style')
        # загружаем фото на обработку
        if (l == 0) :
            bot.send_message(message.from_user.id, "Жду фотку на обработку!")
            fileID = message.photo[-1].file_id
            file_info = bot.get_file(fileID)
            downloaded_file = bot.download_file(file_info.file_path)
            img = io.BytesIO(downloaded_file)
            content_img = image_loader(img, imsize)
            l += 1
            bot.send_message(message.from_user.id, "Енотя доволен!")
        # загружаем стилевое фото
        if (l == 1):
            fileID1 = message.photo[-1].file_id
            if (fileID != fileID1):
                bot.send_message(message.from_user.id, "Енотя очень доволен!")   
                file_info = bot.get_file(fileID1)
                downloaded_file = bot.download_file(file_info.file_path)
                img1 = io.BytesIO(downloaded_file)
                style1_img = image_loader(img1, imsize)
                l += 1
        if l == 1:
            bot.send_message(message.from_user.id, "Теперь загрузи стилевую фотку на обработку!") 
        # обрабатываем
        if l == 2:
            bot.send_message(message.from_user.id, "Енотя полоскает...")
            l = 0
            input_img = content_img.clone()
            # создаем экземпляр нашего класса для обработки с одним стилевым изображением -- и переносим стиль
            st_transf_model = MyStyleModel(input_img, cnn, cnn_normalization_mean, cnn_normalization_std,
                      style1_img, style1_img, content_img, option = 'dual_style', mask_type = None)
            output = st_transf_model.run_style_transfer(num_steps=100)
            # отправляем результат и радуемся)
            img = get_res(output)
            bot.send_photo(message.from_user.id, photo=img)
            bot.send_message(message.from_user.id, "За пушистый хвост енотий поднимем бокальчик!"+"\U0001F942")
            #выводим меню
            signal(message)

# запускаем бота на лонг-поллинге
if __name__ == '__main__':
    bot.polling(none_stop=True, interval=0)
