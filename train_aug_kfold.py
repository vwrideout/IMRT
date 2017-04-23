from training_routine import *
import numpy as np
import vgg_16_keras
from keras.preprocessing import image
import random
from sklearn.model_selection import KFold


#Model pickle will be saved to models/<modelname>
#Performance and predictions will be saved to outfiles/modelname/<example.csv>
modelname = "VGG_pop_dense_Aug_10fold_0415"
ids, y, imgs = get_data()

kf = KFold(n_splits=10, shuffle=True)
k = 1

for train, test in kf.split(imgs):
	with open("log.txt", 'a+') as logfile:
		logfile.write("Starting model " + str(k) + '\n')
	y_test = y[test]
	y_train = y[train]
	imgs_test = imgs[test]
	imgs_train = imgs[train]
	####Data augmentation
	datagen = image.ImageDataGenerator(rotation_range=10, width_shift_range=0.1, height_shift_range=0.1)
	datagen.fit(imgs_train)
	gen_imgs1, gen_y1 = next(datagen.flow(imgs_train, y_train, batch_size=len(imgs_train)))
	gen_imgs2, gen_y2 = next(datagen.flow(imgs_train, y_train, batch_size=len(imgs_train)))
	imgs_train = np.concatenate((imgs_train, gen_imgs1, gen_imgs2))
	y_train = np.concatenate((y_train, gen_y1, gen_y2))
	####
	model = get_model_VGG_pop_dense(False, False)
	history = model.fit(imgs_train, y_train, nb_epoch=300, batch_size=64, validation_data=(imgs_test, y_test)).history
	model.save("models/" + modelname + str(k))
	np.savetxt("outfiles/" + modelname + "/trainlog" + str(k) + ".txt", np.array(history["loss"]), delimiter=',')
	np.savetxt("outfiles/" + modelname + "/valid" + str(k) + ".txt", np.array(history["val_loss"]), delimiter=',')
	pred = [p[0] for p in model.predict(imgs_test)]
	err = y_test - pred
	clipped_pred = [p if p > 0 else 0 for p in pred]
	clipped_err = y_test - clipped_pred
	with open("outfiles/" + modelname + "/results" + str(k) + ".csv", 'w') as f:
	    f.write('rows,id,y,pred,err,cpred,cerr\n')
	    for j in range(len(clipped_pred)):
	        f.write(str(j) + ',' + str(ids[test][j]) + ',' + str(y_test[j]) + ',' + str(pred[j]) + ',' + str(err[j]) + ',' + str(clipped_pred[j]) + ',' + str(clipped_err[j]) + '\n')
	k += 1
