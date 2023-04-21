#从20张SVS文件中随机采样出5800张768*768的图片
#每张SVS文件抽290张 

import openslide
import numpy
import matplotlib.pyplot as plt
import os
import random

svs_path = '/remote-home/share/DATA/RedHouse/AdenocyteBag/SVSfile0413'
files = os.listdir(svs_path)

num = len(files)
i = 0

crop_size = 768
num_crops = 290

for file in files:
    i += 1
    slide = openslide.OpenSlide(os.path.join(svs_path, file))

    level_count = slide.level_count
    print('level_count = ',level_count) 

    [w,h] = slide.dimensions 
    
    for j in range(num_crops):
        
        x = random.randint(w//4, 3*w//4)
        y = random.randint(h//4, 3*h//4)

        tile = numpy.array(slide.read_region((x,y),0, (768, 768)))

        print(i, j)
        plt.imsave("/remote-home/share/DATA/RedHouse/AdenocyteBag/negative_bag_768/{}_{}_{}_negative.png".format(file, x, y), tile)
        
        #plt.figure(figsize=(10, 10))
        #plt.axis('off')
        #plt.imshow(tile)
        #plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
        #plt.margins(0, 0)
        #plt.savefig("C:\\Users\\lonelygod\\Desktop\\normal_pictures\\{}_{}_{}_negative.png".format(file, x, y), dpi = 76.8)

    
