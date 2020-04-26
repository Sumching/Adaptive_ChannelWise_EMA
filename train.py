from solver import GANModel
from dataset import get_loader
import torch
import torchvision
import numpy as np
import sys
if __name__ == "__main__":
    bsize = 1
    z_dim = 128
    model = GANModel(bsize)
    #dataset = create_data('./horse2zebra/', 'train', bsize)
    dataset = get_loader(batch_size=bsize)
    batch_num = len(dataset)
    print(len(dataset))
    model_dir = "./"
    iteration = 0
    while True:
        for idx, data_label in enumerate(dataset):
            #z = torch.randn([bsize, z_dim])
            model.set_input(data_label)
            model.optimize_parametersG()
                #model.optimize_parametersz()
            model.optimize_parametersD()
            
            #if idx % 20 == 0:
            
            #print('\r', '[{}]/[{}] [{}]/[{}] d_loss: {:.6f} g_loss: {:.6f} beta: {:.6f}'.ljust(50).format(epoch, 500, idx, batch_num, model.loss_D, model.loss_G, model.reg_param), end="")
            # torchvision.utils.save_image(model.fake, './ftemp.jpg'.format(epoch), nrow=5, normalize=True)
            # torchvision.utils.save_image(data_label[0], './rtemp.jpg'.format(epoch), nrow=5, normalize=True)
        
                #print("images saved")
            print('\r','[{}]/[{}] d_loss: {:.6f} g_loss: {:.6f}'.format(iteration, 1000000, model.loss_D, model.loss_G))
            model.update_learning_rate()
            if iteration % 50000 == 0:
                torchvision.utils.save_image(data_label[0], './r{}.jpg'.format(iteration+1), nrow=4, normalize=True)
                torchvision.utils.save_image(model.fake, './f{}.jpg'.format(iteration+1), nrow=4, normalize=True)
                torch.save(model.G.state_dict(), model_dir + "{}_G.pth".format(iteration+1))
                torch.save(model.D.state_dict(), model_dir +"{}_D.pth".format(iteration + 1))

            if iteration > 1000000:
                sys.exit('Finish training')
    #            torch.save(model.zmap.state_dict(), model_dir + "{}_zmap.pth".format(epoch + 1))
            iteration += 1

