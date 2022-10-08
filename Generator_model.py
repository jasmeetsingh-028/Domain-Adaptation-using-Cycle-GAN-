import torch
import torch.nn as nn
from torchsummary import summary

class ConvBlock(nn.Module):

    def __init__(self, in_channels, out_channels, down=True, use_act=True, **kwargs):

        super().__init__()

        self.conv= nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                padding_mode="reflect",
                **kwargs
            ) 
            if down
            else nn.ConvTranspose2d(in_channels, 
            out_channels, 
            **kwargs),

            nn.ReLU(inplace=True) if use_act else nn.Identity()

        )
    
    def forward(self,x):

        return self.conv(x)


class ResidualBlock(nn.Module):

    def __init__(self,channels):

        super().__init__()

        self.rblock = nn.Sequential(
            ConvBlock(channels, channels, kernel_size=3, padding=1),
            ConvBlock(channels, channels, use_act = False, kernel_size=3, padding=1)
        )
    
    def forward(self,x):

        return x +self.rblock(x)


class Generator(nn.Module):

    def __init__(self, input_channels=3, feature_num = 64, num_of_residuals=9):

        super().__init__()

        self.initial = nn.Sequential(

            ConvBlock(input_channels, feature_num, kernel_size= 7, stride=1, padding=3),
            ConvBlock(feature_num, feature_num*2, kernel_size= 3, stride=2, padding=1),
            ConvBlock(feature_num*2, feature_num*4, kernel_size= 3, stride=2, padding=1),
        )

        self.residuals = nn.Sequential(
            *[ResidualBlock(feature_num*4) for _ in range(num_of_residuals)]
        )

        self.upSample = nn.ModuleList(
            [
                ConvBlock(feature_num*4, feature_num*2, down= False, kernel_size= 3, stride= 2, padding=1, output_padding=1),
                ConvBlock(feature_num*2, feature_num, down= False, kernel_size= 3, stride= 2, padding=1, output_padding=1),
            ]

            
        )

        self.finalLayer = nn.Sequential(

            nn.Conv2d(
                in_channels = feature_num,
                out_channels= 3, 
                kernel_size = 7,
                stride = 1,
                padding = 3, 
                padding_mode="reflect",
            )

        )

    def forward(self,x):

        x= self.initial(x)
        x= self.residuals(x)

        for layer in self.upSample:
            x= layer(x)
        
        x= self.finalLayer(x)

        return torch.tanh(x)


def main():

    print('imports completed')

    input_tensor = torch.randn(2, 3, 256, 256)

    model= Generator(input_channels=3)

    output_tensor= model(input_tensor)
    
    print(f'Input tensor shape: {input_tensor.shape}')
    print(f'Output shape after feeding the tensor to the model: {output_tensor.shape}')

    summary(model, (3,256,256))

if __name__=="__main__":

    main()




