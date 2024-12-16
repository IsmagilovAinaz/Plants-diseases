import torch
import torch.nn as nn
from torchvision import transforms, datasets
import matplotlib.pyplot as plt
from PIL import Image
import os
from os import listdir
from pathlib import Path

diseases_dict = {0: 'Парша яблони',
1: 'Черная гниль яблони',
2: 'Кедровая яблочная ржавчина',
3: 'Здоровое яблоко',
4: 'Здоровая голубика',
5: 'Здоровая вишня',
6: 'Мучнистая роса вишни',
7: 'Церкоспороз кукурузы',
8: 'Обычновенная ржавчина курурузы',
9: 'Здоровая куруруза',
10: 'Фитофтороз кукурузы',
11: 'Черная гниль винграда',
12: 'Эска винограда', 
13: 'Здоровый виноград', 
14: 'Фитофтороз листьев винограда',
15: 'Болезнь позеленения цитрусовых', 
16: 'Бактериальная пятнистость персика', 
17: 'Здоровый персик',
18: 'Бактериальная пятнистость болгарского перца', 
19: 'Здоровый болгарскиий перец', 
20: 'Ранний фитофтороз картошки',
21: 'Здоровая  картошка', 
22: 'Фитофтороз картошки', 
23: 'Здоровая малина',
24: 'Здоовая соя', 
25: 'Мучистая роса кабачков', 
26: 'Здоровая клубника',
27: 'Ожог листьев клубники', 
28: 'Бактериальная пятнистость помидора', 
29: 'Ранний фитофтороз помидора',
30: 'Здоровый помидор', 
31: 'Фитофтороз помидора', 
32: 'Плесень листьев томата',
33: 'Септориоз листьев томата', 
34: 'Паутинный клещ томата', 
35: 'Целевая пятнистость томата',
36: 'Вирус табачной мозаики томата', 
37: 'Вирус жёлтой курчавости листьев томата'}

model = torch.jit.load('plantsDiseasesModel3.pth')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
model.eval()

print(model)

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.ToTensor()
])

def recognize_image(image_path):
    image = Image.open(image_path)
    image = transform(image).unsqueeze(0)  # Добавляем размерность для батча
    #plt.imshow(image.squeeze().permute(1,2,0))
    #plt.show()
    image = image.to(device)
    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output.data, 1)
    return predicted.item()


folder_dir = r"C:\Users\Ainaz\.cache\kagglehub\datasets\vipoooool\new-plant-diseases-dataset\versions\2\test\test"
folder_dir2 = r"C:\Users\Ainaz\.cache\kagglehub\datasets\vipoooool\new-plant-diseases-dataset\versions\2\New Plant Diseases Dataset(Augmented)\New Plant Diseases Dataset(Augmented)\valid\Potato___healthy"
images = Path(folder_dir).glob('*.jpg')
for image in images:
    dirname, fname = os.path.split(image)
    result = recognize_image(image)
    desease = diseases_dict[result]
    print(f"Дано {fname}. Результат {desease}. {result}")
#result = recognize_image(r"C:\Users\Ainaz\.cache\kagglehub\datasets\vipoooool\new-plant-diseases-dataset\versions\2\test\test\CornCommonRust1.JPG")
#result = recognize_image(r"C:\Users\Ainaz\Desktop\testJPG\1705295732_vsegda-pomnim-com-p-listya-yabloni-kartinki-foto-3.jpg")
#print(f"Predicted class: {result}")