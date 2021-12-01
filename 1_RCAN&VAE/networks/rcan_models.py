import torch.nn as nn
import torch.nn.functional as F
import torch


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


##############################
#           U-NET
##############################


class UNetDown(nn.Module):
    def __init__(self, in_size, out_size, normalize=True, dropout=0.0):
        super(UNetDown, self).__init__()
        layers = [nn.Conv2d(in_size, out_size, 4, 2, 1, bias=False)]
        if normalize:
            layers.append(nn.InstanceNorm2d(out_size))
        layers.append(nn.LeakyReLU(0.2))
        if dropout:
            layers.append(nn.Dropout(dropout))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class UNetUp(nn.Module):
    def __init__(self, in_size, out_size, dropout=0.0, pad=False):
        super(UNetUp, self).__init__()
        if pad:
            layers = [
                nn.ConvTranspose2d(in_size, out_size, 4, 2, 1, bias=False, output_padding=1),
                nn.InstanceNorm2d(out_size),
                nn.ReLU(inplace=True),
            ]
        else:
            layers = [
                nn.ConvTranspose2d(in_size, out_size, 4, 2, 1, bias=False),
                nn.InstanceNorm2d(out_size),
                nn.ReLU(inplace=True),
            ]
        if dropout:
            layers.append(nn.Dropout(dropout))

        self.model = nn.Sequential(*layers)

    def forward(self, x, skip_input):
        x = self.model(x)
        x = torch.cat((x, skip_input), 1)

        return x


class GeneratorUNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=6):
        super(GeneratorUNet, self).__init__()

        self.down1 = UNetDown(in_channels, 64, normalize=False)
        self.down2 = UNetDown(64, 128)
        self.down3 = UNetDown(128, 256)
        self.down4 = UNetDown(256, 512, dropout=0.5)
        self.down5 = UNetDown(512, 512, dropout=0.5)
        self.down6 = UNetDown(512, 512, normalize=False, dropout=0.5)
        # self.down7 = UNetDown(512, 512, dropout=0.5)
        # self.down8 = UNetDown(512, 512, normalize=False, dropout=0.5)

        self.up1 = UNetUp(512, 512, dropout=0.5, pad=True)
        # self.up2 = UNetUp(1024, 512, dropout=0.5)
        # self.up3 = UNetUp(1024, 512, dropout=0.5)
        self.up2 = UNetUp(1024, 512, dropout=0.5, pad=True)
        self.up3 = UNetUp(1024, 256)
        self.up4 = UNetUp(512, 128)
        self.up5 = UNetUp(256, 64)

        self.final_can = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(128, 128, 4, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 3, 3, padding=1),
            nn.Tanh(),
        )

        self.final_seg = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(128, 128, 4, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 10, 3, padding=1),
            nn.Tanh(),
        )

        self.final_dep = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(128, 128, 4, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 1, 3, padding=1),
            nn.Tanh(),
        )



    def forward(self, x):
        # U-Net generator with skip connections from encoder to decoder
        d1 = self.down1(x)
        # print(d1.shape)
        d2 = self.down2(d1)
        # print(d2.shape)
        d3 = self.down3(d2)
        # print(d3.shape)
        d4 = self.down4(d3)
        # print(d4.shape)
        d5 = self.down5(d4)
        # print(d5.shape)
        d6 = self.down6(d5)
        # print(d6.shape)

        # print("Down")
        # d7 = self.down7(d6)
        # d8 = self.down8(d7)

        feature = d6.view(d6.size()[0],-1)
        # print(feature.shape)

        u1 = self.up1(d6, d5)
        # print(u1.shape)
        u2 = self.up2(u1, d4)
        # print(u2.shape)
        u3 = self.up3(u2, d3)
        # print(u3.shape)
        u4 = self.up4(u3, d2)
        # print(u4.shape)
        u5 = self.up5(u4, d1)
        # print(u5.shape)
        # u6 = self.up6(u5, d2)
        # u7 = self.up7(u6, d1)

        return self.final_can(u5), self.final_seg(u5), self.final_dep(u5)


##############################
#        Discriminator
##############################


class Discriminator(nn.Module):
    def __init__(self, in_channels=3, dex = 'can') :
        super(Discriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, normalization=True):
            """Returns downsampling layers of each discriminator block"""
            layers = [nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1)]
            if normalization:
                layers.append(nn.InstanceNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        if dex == 'can':
            in_channels = 3
        elif dex == 'seg':
            in_channels = 10
        elif dex == 'dep':
            in_channels =1

        self.model = nn.Sequential(
            *discriminator_block(in_channels+3 , 64, normalization=False),
            *discriminator_block(64, 128),
            *discriminator_block(128, 256),
            *discriminator_block(256, 512),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(512, 1, 4, padding=1, bias=False)
        )

    def forward(self, img_A, img_B):
        # Concatenate image and condition image by channels to produce input
        img_input = torch.cat((img_A, img_B), 1)
        return self.model(img_input)
