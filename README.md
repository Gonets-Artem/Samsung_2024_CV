# Samsung_2024_CV

## Тема

Тема индивидуального проекта: Замена образа человека в черных очках в видеоконференции

Описание: Если человек желает скрыть свой образ (свое очертание фигуры/лица) из видеоконференции, но при этом хочет оставить задний фон, то для этого достаточно будет надеть на себя черные очки. Задачей проекта является обучение нейросети определять наличие черных очков на лице человека и менять при этом образ человека на его силуэт.


## Тип задачи

- Classification (наличие черных очков в кадре)
- Segmentation (расположение человека и черных очков в кадре)
- Нахождение области пересечения человека и черных очков (чтобы убедиться, что человек сейчас с очками)
- Замена маски человека (силуэта) на черный фон в режиме реального времени


## Данные 

В рамках данной работы было использовано 2 датасета:
– Human segmentation dataset https://www.kaggle.com/datasets/mantasu/glasses-segmentation-synthetic-dataset 
- Glasses segmentation synthetic dataset https://www.kaggle.com/code/mantasu/eyeglasses-classifier 


### 1 Human segmentation dataset 

Первый датасет взят без изменений, в нем содержится 2667 изображений с людьми и их масками для задачи сегментации


### 2 Glasses segmentation synthetic dataset

Второй датасет первоначально содержал более 20 тысяч искусственных фотографий людей со строгим размером 256х256, где они находились в очках разного цвета. После обработки выделены две группы. В первой группе осталось 1961 изображений в черных (очень темных) очках, а также маски для сегментации этих очков. Во второй группе получилось 6543 изображения с нечерными очками, а также маска, содержащая 65536 черных пикселей.
