import torch
import torch.nn as nn
from torch.nn import init
from .base import conv2dbn
from torchvision import models as M

dtype = torch.cuda.FloatTensor


class ConvGRUCell(nn.Module):
    """
    Generate a convolutional GRU cell
    """
    
    def __init__(self, input_size, hidden_size, kernel_size, dtype=torch.FloatTensor):
        super().__init__()
        padding = kernel_size // 2
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.dtype = dtype
        self.reset_gate = nn.Conv2d(input_size + hidden_size, hidden_size, kernel_size, padding=padding)
        self.update_gate = nn.Conv2d(input_size + hidden_size, hidden_size, kernel_size, padding=padding)
        self.out_gate = nn.Conv2d(input_size + hidden_size, hidden_size, kernel_size, padding=padding)
        self.prev_state = None
        
        init.orthogonal_(self.reset_gate.weight)
        init.orthogonal_(self.update_gate.weight)
        init.orthogonal_(self.out_gate.weight)
        init.constant_(self.reset_gate.bias, 0.)
        init.constant_(self.update_gate.bias, 0.)
        init.constant_(self.out_gate.bias, 0.)
    
    def forward(self, input_, prev_state=None):
        
        # get batch and spatial sizes
        batch_size = input_.data.size()[0]
        spatial_size = input_.data.size()[2:]
        
        # generate empty prev_state, if None is provided
        if prev_state is None:
            if self.prev_state is None:
                state_size = [batch_size, self.hidden_size] + list(spatial_size)
                self.prev_state = torch.zeros(state_size, requires_grad=True).type(dtype)
            prev_state = self.prev_state
        
        # data size is [batch, channel, height, width]
        stacked_inputs = torch.cat([input_, prev_state], dim=1)
        update = torch.sigmoid(self.update_gate(stacked_inputs))
        reset = torch.sigmoid(self.reset_gate(stacked_inputs))
        out_inputs = torch.tanh(self.out_gate(torch.cat([input_, prev_state * reset], dim=1)))
        new_state = prev_state * (1 - update) + out_inputs * update
        
        return new_state


class Encoder(nn.Module):
    # Input N x cin x H x w --> N x 512 x H/32 x w/32
    def __init__(self, cin=1):
        super(Encoder, self).__init__()
        pretrained_model = M.resnet34()
        self.model = nn.Sequential(*(
                [nn.Conv2d(cin, 64, 7, 2, padding=3, bias=False)] +
                list(pretrained_model.children())[1:8]))
    
    def forward(self, inputs):
        x = self.model(inputs)
        return x


class Decoder(nn.Module):
    # Input N x 512 x H/32 x W/32 --> N x cout x H x W
    def __init__(self, cout=1):
        super(Decoder, self).__init__()
        self.model = nn.Sequential(*[
            conv2dbn(512, 256, 3, 1),
            nn.Upsample(scale_factor=2, mode='nearest'),
            conv2dbn(256, 128, 3, 1),
            nn.Upsample(scale_factor=2, mode='nearest'),
            conv2dbn(128, 64, 3, 1),
            nn.Upsample(scale_factor=2, mode='nearest'),
            conv2dbn(64, 32, 3, 1),
            nn.Upsample(scale_factor=2, mode='nearest'),
            conv2dbn(32, 16, 3, 1),
            nn.Upsample(scale_factor=2, mode='nearest'),
            conv2dbn(16, 16, 1),
            nn.Conv2d(16, cout, 3, 1, 1)
        ])
    
    def forward(self, inputs):
        x = self.model(inputs)
        return x


class Ensemble(nn.Module):
    def __init__(self, encoder, convgru, decoder):
        super(Ensemble, self).__init__()
        self.encoder = encoder
        self.convgru = convgru
        self.decoder = decoder
    
    def forward(self, input, num_frame, prev_state=None):
        outputs = []
        
        for i in range(num_frame):
            feature = self.encoder(input)
            prev_state = self.convgru(feature, prev_state)
            output = self.decoder(prev_state)
            
            outputs.append(output)
        outputs = torch.cat(outputs, dim=0)
        
        return outputs


if __name__ == '__main__':
    encoder = Encoder()
    decoder = Decoder()
    gru = ConvGRUCell(512, 512, 3, dtype)
    ensemble = Ensemble(encoder, gru, decoder).type(dtype)
    # num_params = sum(np.prod(list(p.size())) for p in ensemble.parameters())
    # print(
    #     'the number of parameter is %f M' % (num_params * 1e-6)
    # )
    input = torch.rand(1, 1, 256, 256).type(dtype)
    out = ensemble(input, 8)
    print(out.shape)
    print(gru.prev_state)

__all__ = [
    "ConvGRUCell",
    "Encoder",
    "Decoder",
    "Ensemble",
]