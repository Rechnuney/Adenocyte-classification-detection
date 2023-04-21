'''
将图像分割成512*512的区域作为bag，根据原标注判断bag的类别, 统计丢掉的bbox数目
'''
import os
import xml.etree.ElementTree as ET
from PIL import Image
import sys

def is_overlap(x1, y1, w1, h1, x2, y2, w2, h2):
    # Check if the two rectangles overlap
    return x1 < x2 + w2 and x1 + w1 > x2 and y1 < y2 + h2 and y1 + h1 > y2

def is_contained(x1, y1, w1, h1, x2, y2, w2, h2):
    # Check if the rectangle 1 is contained in rectangle 2
    return x1 >= x2 and y1 >= y2 and x1 + w1 <= x2 + w2 and y1 + h1 <= y2 + h2

def is_seperated(x1, y1, w1, h1, x2, y2, w2, h2):
    # Check if the two rectangles seperated
    return x1 >= x2 + w2 or x1 + w1 <= x2 or y1 >= y2 + h2 or y1 + h1 <= y2

# Open the image
imgs_path = "/remote-home/share/DATA/RedHouse/Adenocyte/pictures"
xmls_path = '/remote-home/share/DATA/RedHouse/Adenocyte/xmlfile_0118'
imgs_list = os.listdir(imgs_path)
nums = len(imgs_list)
pos = 0
neg = 0

patch_size = 768
overlap = 448 #步长=patch_size-overlap
used_bbox_num = 0 #包括重复使用的bbox
new_used_bbox_num = 0 #不包括重复使用的bbox
drop = 0

for i in range(nums):
    print("第", i, "个文件", imgs_list[i])
    if "png" in imgs_list[i]:
        img_path = os.path.join(imgs_path, imgs_list[i])
    else:
        continue
    xml_path = os.path.join(xmls_path, imgs_list[i][:-4]) + '.xml'
 
    img = Image.open(img_path)
    # Get the size of the image
    width, height = img.size

    # Calculate the number of patches
    num_tiles_x = (width-patch_size) // (patch_size - overlap) + 1 #(4096-512)/(512-128)+1
    num_tiles_y = (height-patch_size) // (patch_size - overlap) + 1
    
    used_bbox = []
    # Iterate over the tiles
    for x in range(num_tiles_x):
        for y in range(num_tiles_y):
            flag_overlap = 0
            # Calculate the boundaries of the tile
            x0 = x * (patch_size-overlap)
            y0 = y * (patch_size-overlap)
            x1 = x0 + patch_size
            y1 = y0 + patch_size
            # Crop the image to the tile
            tile = img.crop((x0, y0, x1, y1))
            tree = ET.parse(xml_path)
            root = tree.getroot()
            masks = root.findall('Mask')    
            patch_masks = [] #一个patch可能包含多个bbox
            for mask in masks:
                bbox = mask.find('bndbox')
                point = bbox.find('LTPoint')
                rect = bbox.find('Rect')
                xmin = int(point.attrib['x'])
                ymin = int(point.attrib['y'])
                h = int(rect.attrib['h'])
                w = int(rect.attrib['w'])
                name = mask.find('name').text.strip()
                
                if is_contained(xmin, ymin, w, h, x0, y0, patch_size, patch_size):
                    if name == 'Pos' or 'AGC' or 'AC':
                        #point.set('x', str(int(point.get('x')) - x0))
                        #point.set('y', str(int(point.get('y')) - y0))
                        patch_masks.append(mask)
                elif is_overlap(xmin, ymin, w, h, x0, y0, patch_size, patch_size):
                    flag_overlap = 1
                    root.remove(mask)
                else:
                    root.remove(mask)
            
            # Save the tile
            if bool(patch_masks):
                pos += 1
                #tree.write('/remote-home/share/DATA/RedHouse/AdenocyteBag/abmil/xmlfile/{}_tile_{}_{}_positve.xml'.format(imgs_list[i],x, y))
                tile.save('/remote-home/share/DATA/RedHouse/AdenocyteBag/abmil_0118data_768_448/{}_tile_{}_{}_positive.jpg'.format(imgs_list[i],x, y))
                for patch_mask in patch_masks:
                    bbox = patch_mask.find('bndbox')
                    point = bbox.find('LTPoint')
                    
                    xmin = int(point.attrib['x'])
                    ymin = int(point.attrib['y'])
                    
                    used_bbox.append([xmin, ymin])
                
            elif flag_overlap == 0 and not patch_masks:
                neg += 1
                tile.save('/remote-home/share/DATA/RedHouse/AdenocyteBag/abmil_0118data_768_448/{}_tile_{}_{}_negative.jpg'.format(imgs_list[i],x, y))

    new_used_bbox = []
    for item in used_bbox:
        if item not in new_used_bbox:
            new_used_bbox.append(item)
    #print(used_bbox,'\n')
    #print(new_used_bbox,'\n')

    tree = ET.parse(xml_path)
    root = tree.getroot()
    all_masks = root.findall('Mask')
    all_bbox = []
    for mask in all_masks:
        bbox = mask.find('bndbox')
        point = bbox.find('LTPoint')
        
        xmin = int(point.attrib['x'])
        ymin = int(point.attrib['y'])
        all_bbox.append([xmin, ymin])

    #print(all_bbox)
    drop += len(all_bbox) - len(new_used_bbox)
    used_bbox_num += len(used_bbox)
    new_used_bbox_num += len(new_used_bbox)
    
    
print('drop:', drop-1)
print('used_bbox_num:', used_bbox_num)
print('new_used_bbox_num:', new_used_bbox_num)
print('pos:', pos)
print('neg:', neg)
