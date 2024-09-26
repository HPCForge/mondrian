import torch
import matplotlib.pyplot as plt

input = torch.load('4_4_input.pt')
pred = torch.load('4_4_pred.pt')
label = torch.load('4_4_label.pt')

print(pred.size())

input = input[16]
pred = pred[16]
label = label[16]

for i in range(len(pred)):
    fig, axarr = plt.subplots(1, 3)
    axarr[0].imshow(torch.rot90(input[i]), vmin=0, vmax=1, cmap='jet')
    axarr[0].set_title('Input')
    axarr[1].imshow(torch.rot90(pred[i]), vmin=0, vmax=1, cmap='jet')
    axarr[1].set_title('Prediction')
    axarr[2].imshow(torch.rot90(label[i]), vmin=0, vmax=1, cmap='jet')
    axarr[2].set_title('Solution')

    axarr[0].axis('off')
    axarr[1].axis('off')
    axarr[2].axis('off')
    plt.tight_layout()
    istr = str(i).zfill(3)
    plt.savefig(f'{istr}.png')
    plt.close()

