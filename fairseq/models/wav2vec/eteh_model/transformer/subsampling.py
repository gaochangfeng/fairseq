#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2019 Shigeki Karita
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Subsampling layer definition."""

import torch

from .embedding import PositionalEncoding


class Conv2dSubsampling(torch.nn.Module):
    """Convolutional 2D subsampling (to 1/4 length).

    Args:
        idim (int): Input dimension.
        odim (int): Output dimension.
        dropout_rate (float): Dropout rate.
        pos_enc (torch.nn.Module): Custom position encoding layer.

    """

    def __init__(self, idim, odim, dropout_rate, pos_enc=None, domain_dim=0):
        """Construct an Conv2dSubsampling object."""
        super(Conv2dSubsampling, self).__init__()
        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(1, odim, 3, 2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(odim, odim, 3, 2),
            torch.nn.ReLU(),
        )
        self.out = torch.nn.Sequential(
            torch.nn.Linear(odim * (((idim - 1) // 2 - 1) // 2) + domain_dim, odim),
            pos_enc if pos_enc is not None else PositionalEncoding(odim, dropout_rate),
        )

    def forward(self, x, x_mask, x_tag=None, offset=0):
        """Subsample x.

        Args:
            x (torch.Tensor): Input tensor (#batch, time, idim).
            x_mask (torch.Tensor): Input mask (#batch, 1, time).

        Returns:
            torch.Tensor: Subsampled tensor (#batch, time', odim),
                where time' = time // 4.
            torch.Tensor: Subsampled mask (#batch, 1, time'),
                where time' = time // 4.

        """
        x = x.unsqueeze(1)  # (b, c, t, f)
        x = self.conv(x)
        b, c, t, f = x.size()
        x = x.transpose(1, 2).contiguous().view(b, t, c * f)
        if x_tag is not None:
            x= torch.cat((x, x_tag.unsqueeze(1).repeat(1,x.size(1),1)), dim=-1)
        if offset == 0:
            x = self.out(x)
        else:
            x = self.out[0](x)
            x = self.out[1](x, offset)
        if x_mask is None:
            return x, None
        return x, x_mask[:, :, :-2:2][:, :, :-2:2]
        
    def forward_nopos(self, x, x_mask):
        x = x.unsqueeze(1)  # (b, c, t, f)
        x = self.conv(x)
        b, c, t, f = x.size()
        x = x.transpose(1, 2).contiguous().view(b, t, c * f)
        x = self.out[0](x)
        if x_mask is None:
            return x, None
        return x, x_mask[:, :, :-2:2][:, :, :-2:2]     

    def forward_addpos(self, xs, offset=0):
        return self.out[1](xs)

    def __getitem__(self, key):
        """Get item.

        When reset_parameters() is called, if use_scaled_pos_enc is used,
            return the positioning encoding.

        """
        if key != -1:
            raise NotImplementedError("Support only `-1` (for `reset_parameters`).")
        return self.out[key]

class Conv2dUpsampling(torch.nn.Module):
    def __init__(self, idim, odim, dropout_rate, pos_enc=None):
        """Construct an Conv2dUpsampling object. 
            When the parameters of the Conv2dUpsampling and Conv2dDownsampling are same,
            the output will has same shape with the input, expect the T dim.
        
        """
        super(Conv2dUpsampling, self).__init__()            
   
        self.odim = odim
        self.idim = idim
        self.middle_dim = ((idim - 1) // 2 - 1) // 2
        
        self.input = torch.nn.Sequential(
            torch.nn.Linear(odim, odim * self.middle_dim),
        )
        
        if idim % 4 ==0:
            p1, p2, o1, o2 = 0, 0, 0, 1 #pad 1 for D dim
        elif idim % 4 == 1:
            p1, p2, o1, o2 = 0, 0, 1, 0 #pad 2 for D dim
        elif idim % 4 == 2:
            p1, p2, o1, o2 = 0, 0, 1, 1 #pad 3 for D dim
        else:
            p1, p2, o1, o2 = 0, 0, 0, 0 #pad 0 for D dim
        
        self.conv = torch.nn.Sequential(
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(odim, odim, 3, 2, padding=0, output_padding=[0, o1]),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(odim, 1, 3, 2, padding=0, output_padding=[0, o2]),
        )     
        
    def forward(self, x, offset=0):
        x = self.input(x)
        b, t, c_f = x.size()
        c, f = self.odim, self.middle_dim
        x = x.view(b, t, c, f).transpose(1,2).contiguous()
        x = self.conv(x).squeeze(1)
        return x

class Conv2dSubsampling6(torch.nn.Module):
    """Convolutional 2D subsampling (to 1/6 length).

    Args:
        idim (int): Input dimension.
        odim (int): Output dimension.
        dropout_rate (float): Dropout rate.
        pos_enc (torch.nn.Module): Custom position encoding layer.

    """

    def __init__(self, idim, odim, dropout_rate, pos_enc=None):
        """Construct an Conv2dSubsampling6 object."""
        super(Conv2dSubsampling6, self).__init__()
        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(1, odim, 3, 2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(odim, odim, 5, 3),
            torch.nn.ReLU(),
        )
        self.out = torch.nn.Sequential(
            torch.nn.Linear(odim * (((idim - 1) // 2 - 2) // 3), odim),
            pos_enc if pos_enc is not None else PositionalEncoding(odim, dropout_rate),
        )

    def forward(self, x, x_mask):
        """Subsample x.

        Args:
            x (torch.Tensor): Input tensor (#batch, time, idim).
            x_mask (torch.Tensor): Input mask (#batch, 1, time).

        Returns:
            torch.Tensor: Subsampled tensor (#batch, time', odim),
                where time' = time // 6.
            torch.Tensor: Subsampled mask (#batch, 1, time'),
                where time' = time // 6.

        """
        x = x.unsqueeze(1)  # (b, c, t, f)
        x = self.conv(x)
        b, c, t, f = x.size()
        x = self.out(x.transpose(1, 2).contiguous().view(b, t, c * f))
        if x_mask is None:
            return x, None
        return x, x_mask[:, :, :-2:2][:, :, :-4:3]


class Conv2dSubsampling8(torch.nn.Module):
    """Convolutional 2D subsampling (to 1/8 length).

    Args:
        idim (int): Input dimension.
        odim (int): Output dimension.
        dropout_rate (float): Dropout rate.
        pos_enc (torch.nn.Module): Custom position encoding layer.

    """

    def __init__(self, idim, odim, dropout_rate, pos_enc=None):
        """Construct an Conv2dSubsampling8 object."""
        super(Conv2dSubsampling8, self).__init__()
        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(1, odim, 3, 2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(odim, odim, 3, 2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(odim, odim, 3, 2),
            torch.nn.ReLU(),
        )
        self.out = torch.nn.Sequential(
            torch.nn.Linear(odim * ((((idim - 1) // 2 - 1) // 2 - 1) // 2), odim),
            pos_enc if pos_enc is not None else PositionalEncoding(odim, dropout_rate),
        )

    def forward(self, x, x_mask):
        """Subsample x.

        Args:
            x (torch.Tensor): Input tensor (#batch, time, idim).
            x_mask (torch.Tensor): Input mask (#batch, 1, time).

        Returns:
            torch.Tensor: Subsampled tensor (#batch, time', odim),
                where time' = time // 8.
            torch.Tensor: Subsampled mask (#batch, 1, time'),
                where time' = time // 8.

        """
        x = x.unsqueeze(1)  # (b, c, t, f)
        x = self.conv(x)
        b, c, t, f = x.size()
        x = self.out(x.transpose(1, 2).contiguous().view(b, t, c * f))
        if x_mask is None:
            return x, None
        return x, x_mask[:, :, :-2:2][:, :, :-2:2][:, :, :-2:2]
