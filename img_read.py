import cv2
import os
import numpy as np
from urllib.request import Request, urlopen


"""
  样本文件夹的创建
  and
  样本图片src_img的导入
"""
def mkdir(path):
  isExist = os.path.exists(path)
  if not isExist:
    os.makedirs(path)
    return True
  else:
    return False


def url_to_img(url):
  hdr = {'User-Agent':'Mozilla/5.0'}
  req = Request(url, headers=hdr)
  response = urlopen(req)
  img_array = np.array(bytearray(response.read()), dtype=np.uint8)
  img = cv2.imdecode(img_array, -1)
  return img

def img_read():
  path = "./test_img/"
  # url = "https://i1.kknews.cc/SIG=28k4ihu/o0r0001rr2418r2rqn4.jpg"
  url = "https://images2017.cnblogs.com/blog/1057546/201708/1057546-20170828110744280-1316246409.jpg"
  mkdir(path)
  src_img = url_to_img(url)
  cv2.imwrite("./test_img/src_img.jpg", src_img)
  # cv2.imshow("src_img", src_img)
