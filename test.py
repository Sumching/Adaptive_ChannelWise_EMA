from solver import GANModel
from dataset import get_loader
import torch
import torchvision
import numpy as np
import cv2
if __name__ == "__main__":
    bsize = 1
    num_imgs = 1900 
    model_dir = "./"
    model = GANModel(bsize)
    model.G.load_state_dict(torch.load(model_dir + "1000001_G.pth"))
    model.D.load_state_dict(torch.load(model_dir + "1000001_D.pth"))
    model.G.eval()
    model.D.eval()
    #model.zmap.load_state_dict(torch.load(model_dir + "20_zmap.pth"))
    #dataset = create_data('./horse2zebra/', 'train', bsize)
    dataset = get_loader(batch_size=bsize)
    data_gen = iter(dataset)

    count = 0
    while count < num_imgs:
        data =next(data_gen)
        model.set_input(data)
        model.forward()
        out = (model.fake.detach().cpu().numpy() + 1)*127.5
        out = np.transpose(out, (0, 2, 3, 1))
        out0 = out
        for idx, img in enumerate(out0):
            cv2.imwrite('./imgs/ex_dir0/{}.jpg'.format(count+idx), img[...,::-1])

        model.set_input(data)
        model.forward()
        out = (model.fake.detach().cpu().numpy() + 1)*127.5
        out = np.transpose(out, (0, 2, 3, 1))
        out1 = out
        for idx, img in enumerate(out1):
            cv2.imwrite('./imgs/ex_dir1/{}.jpg'.format(count+idx), img[...,::-1])

        count += bsize    


            
    # data = next(data_gen)

    # out = [data[0], data[0]]

    # for i in range(16):
    #     model.set_input(data, test=True)
    #     model.forward(test=True)
    #     out.append(model.fake.detach().cpu())

        
        

    # # data = data.cpu().numpy()
    # # fdata = model.fake.detach().cpu().numpy()
    # # np.save('./real.npy', data)
    # # np.save('./freal.npy', fdata)
    # # print("ok!")
    # # model.optimize_parameters()
    
    # out = torch.cat(out, dim=0)
    # torchvision.utils.save_image(out, './test.jpg', nrow=2, normalize=True)

