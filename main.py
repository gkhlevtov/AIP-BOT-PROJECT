import telebot
from telebot import types
from torchvision import models
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from types import SimpleNamespace
import torch
import torchvision
import torch.nn as nn  # здесь лежат все слои
import torchvision.transforms as transforms

bot = telebot.TeleBot(os.environ['TELEGRAM_TOKEN'])


config: SimpleNamespace = SimpleNamespace()  # Создаем базовый класс пространства имен
config.maxSize = 400  # максимально допустимый размер изображения
config.totalStep = 50  # общее количество шагов за эпоху
config.step = 5  # шаг
config.sampleStep = 100  # шаг для сохранения образца
config.styleWeight = 1000  # вес на стиль
config.lr = .003  # шаг обучения
config.content = 'photos\\ny.jpg'
config.style = 'photos\\Style.jpg'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
vgg16 = torchvision.models.vgg16(
    pretrained=True).eval()  # загружаем готовую vgg16 с предобученными весами, переключаем в режим проверки

user_dict = dict()

start_markup = types.ReplyKeyboardMarkup(resize_keyboard=True)
ready_markup = types.ReplyKeyboardMarkup(resize_keyboard=True)
item1 = types.KeyboardButton('🎮 Начать')
item2 = types.KeyboardButton('💡 Общая информация')
item3 = types.KeyboardButton('✅ Обработать фото')
item4 = types.KeyboardButton('ℹ️ Проверить фото')
item5 = types.KeyboardButton('1️⃣ Сменить оригинальное фото')
item6 = types.KeyboardButton('2️⃣ Сменить фото стиля')

start_markup.add(item1, item2)
ready_markup.add(item3, item4, item5, item6)


class UserInfo:
    """Класс для хранения информации о пользователе"""

    def __init__(self, userid: str, original_photo: bool, style_photo: bool, config):
        self.id = userid
        self.original_photo = original_photo
        self.style_photo = style_photo
        self.content = config.content
        self.style = config.style
        self.directory = f'users\\user{self.id}'


class PretrainedNet(nn.Module):
    """Класс для обозначения предобученной сети"""

    def __init__(self):
        # Инициализирую модель
        super(PretrainedNet, self).__init__()
        self.select = [0, 5, 7, 10, 15]  # те слои, через которые я буду пропускать изображение
        self.pretrainedNet = models.vgg19(pretrained=True).to(device)  # подгружаю предобученную сеть

    def forward(self, x):
        features = []  # Извлекаю по индексам, которые я прописал выше, feature map
        output = x
        for layerIndex in range(len(self.pretrainedNet.features)):
            output = self.pretrainedNet.features[layerIndex](output)
            if layerIndex in self.select:
                features.append(output)
        return features


def load_image(image_path, transform=None, max_size=None, shape=None):
    """Функция загрузки изображения"""
    # Загружаем изображение
    image = Image.open(image_path)

    # Если указан максимальный размер, то меняем размер нашего изображения
    if max_size:
        scale = max_size / max(image.size)  # задаем масштаб для преобразования размера
        size = np.array(image.size) * scale  # масштабированный размер
        image = image.resize(size.astype(int), Image.ANTIALIAS)  # преобразуем

    # Если указана форма изображением, меняем форму
    if shape:
        image = image.resize(shape, Image.LANCZOS)

    # Если указаны методы трансформирования, то применяем его
    if transform:
        image = transform(image).unsqueeze(0)  # трансформировали + вытянули до батча

    return image.to(device)


def inference(target, user: UserInfo):
    """Фунция для сохранения фото после обработки"""
    inv_normalize = transforms.Normalize(mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.255],
                                         std=[1 / 0.229, 1 / 0.224, 1 / 0.255])
    inv_content = inv_normalize(target[0])

    styled_img_mp = inv_content.cpu().detach().numpy().transpose(1, 2, 0)
    styled_img_mp[styled_img_mp > 1.] = 1.
    styled_img_mp[styled_img_mp < 0] = 0.

    plt.imshow(styled_img_mp)
    plt.gca().set_axis_off()
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0,
                        hspace=0, wspace=0)
    plt.margins(0, 0)
    plt.savefig(f"{user.directory}\\result_image.png", bbox_inches='tight', pad_inches=0)
    plt.clf()

    return styled_img_mp


@bot.message_handler(commands=['start'])
def send_welcome(message):
    """Функция для отправки привествия при первом запуске"""
    global start_markup
    chat_id = message.chat.id
    user = UserInfo(chat_id, False, False, config)
    user_dict.update({user.id: user})
    try:
        os.mkdir(user_dict[chat_id].directory)
    except FileExistsError:
        print('Данный пользователь уже существует')
    finally:
        bot.reply_to(message,
                     f'Привет, {message.from_user.first_name}! Я умею обрабатывать фото по стилю. '
                     'Чтобы начать, нажми на кнопку снизу',
                     reply_markup=start_markup)


@bot.message_handler(content_types=['text'])
def bot_message(message):
    if message.text == '🎮 Начать':
        begin_message(message)

    if message.text == '💡 Общая информация':
        bot.send_message(chat_id=message.chat.id, text='Этот бот был выполнен в рамках проекта, '
                                                       'принцип его работы очень прост:')
        bot.send_photo(chat_id=message.chat.id, photo=open('photos\\info.jpg', 'rb'), reply_markup=start_markup)

    if message.text == '✅ Обработать фото':
        do_style(message)

    if message.text == 'ℹ️ Проверить фото':
        check_photos(message)

    if message.text == '1️⃣ Сменить оригинальное фото':
        change_original_photo(message)

    if message.text == '2️⃣ Сменить фото стиля':
        change_style_photo(message)


@bot.message_handler(commands=['begin'])
def begin_message(message):
    bot.reply_to(message, "Отправьте два фото, сначала оригинальное, затем фото стиля")


@bot.message_handler(content_types=['photo'])
def photo(message):
    chat_id = int(message.chat.id)
    user = user_dict[chat_id]
    original_photo = user.original_photo
    style_photo = user.style_photo

    fileid = message.photo[-1].file_id
    file_info = bot.get_file(fileid)
    downloaded_file = bot.download_file(file_info.file_path)

    if not original_photo:
        with open(f"{user.directory}\\original_image.jpg", 'wb') as new_file:
            new_file.write(downloaded_file)
            new_file.close()

        user.original_photo = True
        user.content = f"{user.directory}\\original_image.jpg"

        bot.reply_to(message, "Оригинальное фото готово", reply_markup=ready_markup)

        if original_photo and style_photo:
            bot.send_message(chat_id=message.chat.id,
                             text='Все фото готовы, нажмите на соответствующую кнопку,'
                                  ' в зависимости от того, что хотите сделать',
                             reply_markup=ready_markup)

    elif original_photo and not style_photo:
        with open(f"{user.directory}\\style_image.jpg", 'wb') as new_file:
            new_file.write(downloaded_file)
            new_file.close()

        user.style_photo = True
        user.style = f"{user.directory}\\style_image.jpg"

        bot.reply_to(message, "Фото стиля готово", reply_markup=ready_markup)

        if original_photo and style_photo:
            bot.send_message(chat_id=message.chat.id,
                             text='Все фото готовы, нажмите на соответствующую кнопку,'
                                  ' в зависимости от того, что хотите сделать',
                             reply_markup=ready_markup)

    else:
        bot.send_message(chat_id=message.chat.id,
                         text='Все фото готовы, нажмите на соответствующую кнопку,'
                              ' в зависимости от того, что хотите сделать',
                         reply_markup=ready_markup)

    user_dict.update({user.id: user})


@bot.message_handler(commands=['switch_original'])
def change_original_photo(message):
    """Функция для смены оригинального фото"""
    chat_id = int(message.chat.id)
    user = user_dict[chat_id]

    user.content = 'photos\\ny.jpg'
    user.original_photo = False
    user_dict.update({user.id: user})

    bot.reply_to(message, "Для смены оригинального фото отправьте его снова")


@bot.message_handler(commands=['switch_style'])
def change_style_photo(message):
    """Функция для смены фото стиля"""
    chat_id = int(message.chat.id)
    user = user_dict[chat_id]

    user.style = 'photos\\Style.jpg'
    user.style_photo = False
    user_dict.update({user.id: user})

    bot.reply_to(message, "Для смены фото стиля отправьте его снова")


@bot.message_handler(commands=['check'])
def check_photos(message):
    """Функция для проверки фото"""
    chat_id = int(message.chat.id)
    user = user_dict[chat_id]
    original_photo = user.original_photo
    style_photo = user.style_photo

    if not original_photo and not style_photo:
        bot.reply_to(message, "У Вас нет готовых фото")

    elif original_photo and not style_photo:
        bot.reply_to(message, "У Вас готово только оригинальное фото")
        bot.send_photo(chat_id=message.chat.id, photo=open(f"{user.directory}\\original_image.jpg", 'rb'))

    elif not original_photo and style_photo:
        bot.reply_to(message, "У Вас готово только фото стиля:")
        bot.send_photo(chat_id=message.chat.id, photo=open(f"{user.directory}\\style_image.jpg", 'rb'))

    else:
        bot.send_message(chat_id=message.chat.id,
                         text='Все фото готовы, нажмите на соответствующую кнопку,'
                              ' в зависимости от того, что хотите сделать',
                         reply_markup=ready_markup)
        bot.send_message(chat_id=message.chat.id, text='Ваши фото:')
        bot.send_photo(chat_id=message.chat.id, photo=open(f"{user.directory}\\original_image.jpg", 'rb'))
        bot.send_photo(chat_id=message.chat.id, photo=open(f"{user.directory}\\style_image.jpg", 'rb'))

    user_dict.update({user.id: user})


@bot.message_handler(commands=['do_it'])
def do_style(message):
    """Функция для конечной обработки фото с помощью НС"""
    global config
    chat_id = int(message.chat.id)
    user = user_dict[chat_id]
    original_photo = user.original_photo
    style_photo = user.style_photo
    config.content = user.content
    config.style = user.style

    if original_photo and style_photo:
        bot.send_message(chat_id=message.chat.id, text='Приступаю к обработке фото, это займёт некоторое время...')

        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                                                                    std=(0.229, 0.224, 0.225))])

        content = load_image(config.content, transform, max_size=config.maxSize)
        style = load_image(config.style, transform, shape=[content.size(3), content.size(2)])
        target = content.clone().requires_grad_(True)

        model = PretrainedNet().eval()  # для использования весов предобученной сетки переводим ее в режим eval
        optimizer = torch.optim.Adam([target],
                                     lr=0.1)
        content_criteria = nn.MSELoss()

        for step in range(config.totalStep):
            # Для каждого из изображений извлекаем feature map
            target_features = model.forward(target)
            content_features = model.forward(content)
            style_features = model.forward(style)

            style_loss = 0
            content_loss = 0

            for f1, f2, f3 in zip(target_features, content_features, style_features):
                # Вычисляем потери для оригинала и конечной картинки
                content_loss += content_criteria(f1, f2)

                # Меняем форму сверточных feature maps. Приводим к формату (количество каналов, ширина*высота)
                _, c, h, w = f1.size()  # пропускаем batch
                f1 = f1.reshape(c, h * w).to(device)
                f3 = f3.reshape(c, h * w).to(device)

                # Находим матрицу Грама для конечной и стиля
                f1 = torch.mm(f1, f1.t())
                f3 = torch.mm(f3, f3.t())

                # Потери для стиля и конечной картинки
                # kf1 = 1 / (4 * (len(f1) * len(f3)) ** 2)
                kf2 = 1 / 4 * (len(f1) * len(f3)) ** 2
                # kf3 = 1 / (c * w * h)
                style_loss += content_criteria(f1, f3) * kf2

            # Прописываем конечную функцию потерь
            loss = style_loss + content_loss
            # print(betta)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (step + 1) % config.step == 0:
                print('Шаг [{}/{}], Ошибка для оригинала: {:.4f}, Ошибка для стиля: {}'
                      .format(step + 1, config.totalStep, content_loss.item(), style_loss.item()))

            if (step + 1) % config.sampleStep == 0:  # сохраняем нашу картинку
                img = target.clone().squeeze()  # создаем место под тензор
                img = img.clamp_(0, 1)  # оставить значения, попадающие в диапазон между 0,1
                torchvision.utils.save_image(img, 'output-{}.png'.format(step + 1))

        inference(target, user)

        user.original_photo = False
        user.style_photo = False
        os.remove(f"{user.directory}\\original_image.jpg")
        os.remove(f"{user.directory}\\style_image.jpg")
        user.content = 'photos\\ny.jpg'
        user.style = 'photos\\Style.jpg'

        bot.send_message(chat_id=message.chat.id, text='Ваше фото готово:', reply_markup=start_markup)
        bot.send_photo(chat_id=message.chat.id, photo=open(f"{user.directory}\\result_image.png", 'rb'))
        os.remove(f"{user.directory}\\result_image.png")

    else:
        bot.reply_to(message, 'Не все фото готовы, нажмите "ℹ️ Проверить фото", чтобы посмотреть текущие фото')

    user_dict.update({user.id: user})


bot.infinity_polling()
