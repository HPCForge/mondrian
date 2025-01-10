# FactFormer

The code in this directoy is taken from the [FactFormer](https://github.com/BaratiLab/FactFormer/tree/main) github.
It is essentially unmodified, apart from reformatting.

The only notable change is to the API, so I pass in the domain x/y size
to `FactFormer2D`, rather than passing in the x-coordinates and y-coordinates.
This is just for consistency with the other models' APIs.