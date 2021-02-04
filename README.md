# Enot_for_Heroku

# Telebot
Асинхронный Телебот: @Enotuskabot

***

У нас есть асинхронный бот @Enotuskabot(файл bot.py), построенный с помощью фреймворка Telebot  и осуществляющий перенос стиля и улучшение фотографий.

У бота также есть дополнительная команда "Узнай у ученого Еноти", благодаря которой пользователь может задать ученому Боту Еноту любой вопрос и получить 3 самые актуальные  ссылки с ответами.

Пока осуществляется перенос стиля, можно читать информацию по этим ссылкам и самообразовываться :))

Бот может переносить стиль  следующими способами:

1. Перенос одного стиля на изображение

2. Перенос двух стилей с использованием маски -- доступно 4 типа масок (см.папку Masks)

3. Одновременный перенос двух стилей на изображение.

В этой части проекта я использую собственный нейросетевой класс  (см. файл styletransfer.py; базовая сеть -- VGG-19)

Также бот умеет применять к картинкам, которые он предварительно сжимает для более наглядного результата, операцию Супер Резолюции.
Здесь я использовала готовую предобученную сеть  на Tensor Flow из проекта https://github.com/idealo/image-super-resolution

В проект для удобства тестирования добавлен ноутбук Enotya_bot.inpyb, который легко можно запустить на Google Colab -- так на  GPU, так и на CPU. 

Модель изначально отлаживалась для работы на CPU: бот обрабатывает картинки размером 256 на 256 в течение 4 мин. (128 на 128 -- за 2 мин); с GPU он работает практически мгновенно.

Бот запущен на Heroku, но работает там небыстро. я ы рекомендовала тестировать локально / по ссылке на Colab Notebook

Проект докеризован и выложен на DockerHub: nastyakrass/enotyar. 

Зеленые галочки в левом вернхем углу, которые видны при просмотре файлов на Гитхабе, означают, что проект прошел проверку Докером и успешно собран

