import torch
import torch.nn as nn
import torch.nn.functional as F

class NeRF(nn.Module):
    def __init__(self, D=8, W=256, input_ch=63, input_ch_views=27, output_ch=4, skips=[4]):
        """
        D: Numarul de straturi (profunzime)
        W: Latimea fiecarui strat (numar de neuroni)
        input_ch: Dimensiunea XYZ dupa Positional Encoding (3 + 3*2*10 = 63)
        input_ch_views: Dimensiunea directiei dupa Positional Encoding (3 + 3*2*4 = 27)
        skips: Indexul stratului unde reintroducem input-ul (Residual connection)
        """
        super(NeRF, self).__init__()
        self.D = D
        self.W = W
        self.input_ch = input_ch
        self.input_ch_views = input_ch_views
        self.skips = skips
        
        # Straturile care proceseaza coordonatele spatiale (XYZ)
        self.pts_linears = nn.ModuleList(
            [nn.Linear(input_ch, W)] + 
            [nn.Linear(W, W) if i not in self.skips else nn.Linear(W + input_ch, W) for i in range(D-1)]
        )
        
        # Stratul care extrage densitatea (Sigma) - depinde doar de XYZ
        self.alpha_linear = nn.Linear(W, 1)
        
        # Straturile care proceseaza culoarea (RGB) - depind de XYZ si de directia vizualizarii
        self.feature_linear = nn.Linear(W, W)
        self.views_linears = nn.ModuleList([nn.Linear(input_ch_views + W, W//2)])
        self.rgb_linear = nn.Linear(W//2, 3)

    def forward(self, x):
        # Despartim input-ul concatenat inapoi in Pozitie (XYZ) si Directie (Views)
        inputs, views = torch.split(x, [self.input_ch, self.input_ch_views], dim=-1)
        
        h = inputs
        for i, l in enumerate(self.pts_linears):
            h = self.pts_linears[i](h)
            h = F.relu(h)
            # Adaugam skip-connection pentru a nu pierde detaliile spatiale
            if i in self.skips:
                h = torch.cat([inputs, h], -1)

        # Sigma (densitatea) trebuie sa fie independenta de unghiul din care privim
        sigma = self.alpha_linear(h)
        feature = self.feature_linear(h)
        
        # Culoarea (RGB) depinde de unghi (pentru a invata reflexii si luciu)
        h = torch.cat([feature, views], -1)
        for i, l in enumerate(self.views_linears):
            h = self.views_linears[i](h)
            h = F.relu(h)

        rgb = torch.sigmoid(self.rgb_linear(h))
        
        # Returnam [R, G, B, Sigma]
        return torch.cat([rgb, sigma], -1)