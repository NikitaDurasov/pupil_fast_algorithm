from PIL import Image
import numpy as np
import pandas as pd
import posixpath
import matplotlib.pyplot as plt
import imtools

parsed_coord = open(posixpath.abspath('parsed_data/params_res.txt'))
parsed_coord = parsed_coord.readlines()

res = []
for line in parsed_coord:
    res.append(line.split())
df = pd.DataFrame(res)

image_names = imtools.get_imlist(posixpath.abspath('parsed_data/imgs'))

image = Image.open(image_names[2])
image_arr = np.array(image)

plt.figure(1)
plt.imshow(image_arr)
plt.plot(df[df['Filename'] == image_names[2]].CenX, df[df['Filename'] == image_names[2]].CenY, 'r*')
plt.show()

