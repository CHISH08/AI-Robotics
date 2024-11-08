# AI-Robotics

## О чем репозиторий

В данном репозитории хотелось бы рассказать о своем первом и увлекательном опыте внедрения искусственного интеллекта для автоматизации движения и выполнений действий робота.

## В чем состояла задача

В данном файле представлен регламент хакатона. 

[регламент](<data_for_present/Регламент Yandex Camp v3.pdf>)

Если вкратце: нужно переносить объекты в корзинку и нажать на кнопку единожды. Также робот должен передвигаться неврезаясь в стенки и другие объекты.

## Моя роль в команде

Так как в моей команде не было ни одного человека, работавшего с компьютерным зрением, данную ношу взял на себя я, оставив для других задачу механики и управления робота. Таким образом, моей задачей было:
- fine-tuning YOLO
- Выделение статического поля игры со стенами
- Определение открытых стенок в ящике, в котором лежит шарик
- Детекция объектов с верхней камеры
- Детекция объектов с нижней камеры

Про особенность каждой из задачь я расскажу далее.

## fine-tuning YOLO

Т.к. нужно было работать с видео в реальном времени, а также на выданных нам ноутбуках не было gpu, пришлось брать yolo v5 nano от ultralitics. Она удобна тем, что имеет маленькое количество параметров, и потому она отрабатывала на cpu всего за 0.08 секунды (8 милисекунд), что почти не ощущается при использовании в реальном времени.

Для тонкой настройки модели я долго искал оптимальную аугментацию для учета всех возможных случаев (и для тени и для света), тк было всего 1 обработанное видео заезда и не хотелось делать большой датасет. Затем я заморозил все слои, кроме 9 последних и начал обучать.

При таком количестве данных, модель находила объекты везде: в тени, в свету, видя только часть и т.п.

Код fine-тюнинга:
[fine-tune](finetune_yolo.ipynb)

Метрики:
<p align="center">
  <img src="data_for_present/confusion_matrix.png" width="34%" />
  <img src="data_for_present/F1_curve.png" width="38.5%" />
  <img src="data_for_present/labels.jpg" width="26%" />
</p>

## Выделение статического поля игры со стенами

Тк поле было достаточно простое и имело одни и теже объекты, мое решение опиралось на нескольких гипотезах

- Изначально нам дано, что камера искривленно снимает, потому ее надо выравнять
- Все объекты, имеющие высокий цветовой градиент - не пол. Таким образом можно было без проблем с высокой скоростью выделять стены и зону движения.
- Самый большой объект на поле - ящик.
- Открытые стенки ящика имеют больший процент содержания "светлых" пикселей. Таких стенок всего две из четырех и они всегда на противоположных сторонах
- Все объекты, которые нас интересуют - не статичны, потому их можно исключать из поля зрения

Таким образом, приняты были следующие меры:

- Применение дисторсии для исправления закруглений во время съемки
- Применение чб-фильтра для нахождения наиболее черных и выделяющихся среди основного поля объектов
- После применения фильтра проходиться небольшим ядром и на основе его зачищать от белых пикселей места, где была темень и тому подобное
- Пуск в начале программы большого ядра, которое единожды проходится по полю и ищет пиксель, принадлежащий ящику
- От пикселя, принадлежащего ящику пускалось небольшое ядро, которое, на основе процента белых пикселей, искало края ящика. Если процент меньше 1/4, значит это край (на углу мы получаем что заполнен только определенный край вадрата из четырех)
- От крайних точек ящика пускаем по прямой линие небольшое ядро, считающее среднее кол-во белых пикселей на стороне. Если на определенных двух противоположных сторонах среднее больше, значит там есть стенки, и потому туда робот не проедет. Иначе - там нет стенок
- После детекции была произведена зачистка поля от нестатичных объектов

Итоги:

<video controls src="data_for_present/static_top.mp4" title="Title"></video>

## Определение открытых стенок в ящике, в котором лежит шарик

Для нахождения открытых стенок в коробке:

1) Находим коробку
2) Ищем ее границы, те последние белые пиксели, с помощью небольших ядер, которые ищут момент, когда кол-во белых пикселей в ядре меньше 30 процентов, тк на границе белые пиксели должны быть в правой нижней части ядра
3) пускаем небольшие ядра от границ со всех сторон
4) если белых пикселей суммарно в левой и правой больше, чем в нижней и верхней, то левая и правая стенки закрыты, иначе наоборот
5) левая и правая стенка обозначаются как 16 и 17 соответственно (в дальнейшем это будет использоваться), а нижняя и верхняя как 14, 19

Итоги:
[find_boxes](find_open_boxes.ipynb)

## Детекция объектов с верхней камеры

Для данной части:

1) Рисуем статическое поле
2) Детектируем интересующие нас объекты
3) Покрываем маску статического поля пикселями интересующих нас объектов

Итоги:
<video controls src="data_for_present/dinamic_top.mp4" title="Title"></video>

## Детекция объектов с нижней камеры

Полностью аналогично детекции с верхней камеры, вот как это выглядит:

<video controls src="data_for_present/down_camera.mp4" title="Title"></video>

## Подытожим

Для чего все это пишется? Пишется для того, чтобы показать насколько мощны алгоритмы компьютерного зрения в руках думающего человека. Вы только посмотрите, любую информацию о границах объекта, его классе можно узнать, даже не используя сложные нейронные сети. Мне очень понравился данный опыт, тк я не просто смотрел на метрики, я видел реальные результаты при движении робота и понимал, что стоит подправить. Учитывая, что я не занимался почти год компьютерным зрением, могу сказать, что время провел с пользой и очень классно!
