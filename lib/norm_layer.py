import torch

class NormLayer(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def norm_image(self, img):
        LS = len(img.shape)
        EPS = 1e-8
        img = img.float()
        if LS==3:
            C = img.shape[0]
            for c in range(C):
                imgc = img[c,:,:]
                mn = torch.mean(imgc)
                ss = torch.std(imgc, unbiased=True) + EPS
                img[c,:,:] = (imgc - mn)/ss
        else:
            mn = torch.mean(img)
            ss = torch.std(img, unbiased=True) + EPS
            img = (img - mn) / ss
        return img
    def forward(self,ten):
        N = ten.shape[0]
        for n in range(N):
            tn = ten[n,:]
            tn = self.norm_image(tn)
            ten[n,:] = tn
        return ten


if __name__ == '__main__':
    sz = (3,2,4)
    ten1 = torch.normal(2, 3, size=sz) * 5
    ten2 = torch.normal(2, 3, size=sz) * 10
    sp = (2,) + sz
    tens = torch.zeros(sp)
    tens[0,] = ten1
    tens[1,] = ten2
    nl = NormLayer()
    tens = nl.forward(tens)