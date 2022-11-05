# source: https://www.askpython.com/python/examples/mat-files-in-python
from scipy.io import loadmat
import pandas as pd


files = "/media/andres/2D2DA2454B8413B5/software_proj/hico_20160224_det/"
annotations_1 = files + "anno.mat"
annotations_2 = files + "anno_bbox.mat"

group_1 = loadmat(annotations_1)
group_2 = loadmat(annotations_2)
headers_group_1 = [el for el in group_1]  # list_action, anno_train, anno_test, list_train, list_test
# anno_train and anno_test have a bunch of nans
# list_test and list_train have the exact file names
# list action seems to list all of the words?
headers_group_2 = [el for el in group_2]  # bbox_train, bbox_test, list_action
group_1_list_act = [[el for el in upperel] for upperel in group_1["list_action"]]
#print(len(group_1_list_act))  # 600
# train will yield 38118
#print(group_1_list_act[3])

#[(array(['airplane'], dtype='<U8'), array(['fly'], dtype='<U3'), array(['flying'], dtype='<U6'), array(['fly, aviate, pilot'], dtype='<U18'), array(['operate an airplane'], dtype='<U19'), array([[(array([[3]], dtype=uint8), array(['v01941093'], dtype='<U9'), array(['fly.v.03'], dtype='<U8'), array(['6'], dtype='<U1'), array(['fly aviate pilot'], dtype='<U16'), array(['operate an airplane'], dtype='<U19'), array(['The pilot flew to Cuba'], dtype='<U22'))]],
      #dtype=[('id', 'O'), ('wid', 'O'), ('name', 'O'), ('count', 'O'), ('syn', 'O'), ('def', 'O'), ('ex', 'O')]), array([], dtype='<U1'))]


# in group 2 bbox must be flattened
group_2_bbox_train = [el for upperel in group_2["bbox_train"] for el in upperel]

#(array(['HICO_train2015_00000006.jpg'], dtype='<U27'), array([[(array([[640]], dtype=uint16), array([[457]], dtype=uint16), array([[3]], dtype=uint8))]],
#      dtype=[('width', 'O'), ('height', 'O'), ('depth', 'O')]), array([[(array([[578]], dtype=uint16), array([[(array([[76]], dtype=uint8), array([[281]], dtype=uint16), array([[67]], dtype=uint8), array([[451]], dtype=uint16))]],
#              dtype=[('x1', 'O'), ('x2', 'O'), ('y1', 'O'), ('y2', 'O')]), array([[(array([[3]], dtype=uint8), array([[382]], dtype=uint16), array([[20]], dtype=uint8), array([[250]], dtype=uint8))]],
#              dtype=[('x1', 'O'), ('x2', 'O'), ('y1', 'O'), ('y2', 'O')]), array([[1, 1]], dtype=int32), array([[0]], dtype=uint8))                                                                     ,
#        (array([[583]], dtype=uint16), array([[(array([[72]], dtype=uint8), array([[288]], dtype=uint16), array([[84]], dtype=uint8), array([[456]], dtype=uint16))]],
#              dtype=[('x1', 'O'), ('x2', 'O'), ('y1', 'O'), ('y2', 'O')]), array([[(array([[8]], dtype=uint8), array([[372]], dtype=uint16), array([[16]], dtype=uint8), array([[334]], dtype=uint16))]],
#              dtype=[('x1', 'O'), ('x2', 'O'), ('y1', 'O'), ('y2', 'O')]), array([[1, 1]], dtype=int32), array([[0]], dtype=uint8))                                                                      ]],
#      dtype=[('id', 'O'), ('bboxhuman', 'O'), ('bboxobject', 'O'), ('connection', 'O'), ('invis', 'O')]))


#group_2_bbox_train = [el for upperel in group_2["bbox_train"] for el in upperel]
group_2_list_act = [[el for el in upperel] for upperel in group_2["list_action"]]
#print(group_2_list_act[5]) structure similar to other list_action


#%% let's work with image number 6 as an example again
group_1_list_act = [[el for el in upperel] for upperel in group_1["list_action"]]
group_1_list_train = [[el for el in upperel] for upperel in group_1["list_train"]]
group_1_list_test = [[el for el in upperel] for upperel in group_1["list_test"]]
group_1_anno_train = [[el for el in upperel] for upperel in group_1["anno_train"]]
print(group_1_list_act[5])
print("-------------------\n\n")
print(group_1_list_train[5])
print("-------------------\n\n")
print(group_1_list_test[5])
print("-------------------\n\n")
#print(group_1_anno_train[5]) NANS, why
print("-------------------\n\n")
print(group_1["anno_test"][5])
# pressume that anno_test will be nans as well
