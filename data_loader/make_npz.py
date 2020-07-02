import numpy as np
import os,re,cv2

def load_file_list(path, regx):
    file_list = os.listdir(path)
    return_list = []
    for _, f in enumerate(file_list):
        if re.search(regx, f):
            return_list.append(os.path.join(path,f))
    return sorted(return_list)

def read_image(path_list):
    rslt = []
    for path in path_list:
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)[:, :, 0]
        img = np.expand_dims(img, axis = 2)
        rslt.append(img)
    return np.array(rslt)

def read_salmap(path_list):
    rslt = []
    for path in path_list:
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = np.expand_dims(img, axis = 2)
        rslt.append(img)
    return np.array(rslt)

def save_npz(datapath, savepath):
    t_hr_path = os.path.join(datapath, 'train', 'HR')
    t_sal_map_path = os.path.join(datapath, 'train', 'Saliency')
    v_hr_path = os.path.join(datapath, 'valid', 'HR')
    v_sal_map_path = os.path.join(datapath, 'valid', 'Saliency')

    t_names = load_file_list(t_hr_path, regx='.*.png')
    t_names = [os.path.basename(n) for n in t_names]
    t_hr = read_image(load_file_list(t_hr_path, regx='.*.png'))
    t_sal_map = read_salmap(load_file_list(t_sal_map_path, regx='.*.png'))

    v_names = load_file_list(v_hr_path, regx='.*.png')
    v_names = [os.path.basename(n) for n in v_names]
    v_hr = read_image(load_file_list(v_hr_path, regx='.*.png'))
    v_sal_map = read_salmap(load_file_list(v_sal_map_path, regx='.*.png'))

    train = {
        'names' : t_names,
        'hr' : t_hr,
        'sal_map' : t_sal_map
    }
    valid = {
        'names' : v_names,
        'hr' : v_hr,
        'sal_map' : v_sal_map
    }

    np.savez_compressed(savepath,
            train = train,
            valid = valid,
            )

datapath = '../data/tgrs-va-fbgan/'
savepath = 'data/sr_geo'
save_npz(datapath, savepath)