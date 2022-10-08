import torch      
import torch.nn as nn
from torchsummary import summary


class Block(nn.Module):
    def __init__(self, input_channels, output_channels, stride):

        super().__init__()

        self.conv= nn.Sequential(

            nn.Conv2d(
                input_channels, 
                output_channels,
                kernel_size = 4, 
                stride = stride, 
                padding = 1, 
                bias=True, 
                padding_mode="reflect"),

            nn.InstanceNorm2d(output_channels),
            nn.LeakyReLU(0.2),
        )

    def forward(self,x):
        
        return self.conv(x)


class Discriminator(nn.Module):
    def __init__(self, input_channels=3, features=[64, 128, 256, 518]): 

        super().__init__()
        
        self.initial = nn.Sequential(

            nn.Conv2d(
                input_channels,
                features[0],
                kernel_size=4,
                stride=2,
                padding=1,
                padding_mode="reflect",
            ),

            nn.LeakyReLU(0.2)

        )

        layers=[]

        for i in range(len(features[:-1])):

            layers.append(Block(features[i], features[i+1], stride= 1 if features[i+1]==features[-1] else 2))
            in_channel=features[i+1]
        
        layers.append(nn.Conv2d(

            in_channels = in_channel,
            out_channels = 1,
            kernel_size = 4,
            stride = 1,
            padding = 1,
            padding_mode = "reflect",
        ))

        self.model= nn.Sequential(*layers)

    def forward(self,x):

        x = self.initial(x)

        x = self.model(x)

        return torch.sigmoid(x)


def main():

    print('imports completed')

    input_tensor = torch.randn(5, 3, 256, 256)

    model= Discriminator()

    output_tensor= model(input_tensor)
    
    print(f'Input tensor shape: {input_tensor.shape}')
    print(f'Output shape after feeding the tensor to the model: {output_tensor.shape}')

    summary(model, (3,256,256))

if __name__=="__main__":

    main()


    

