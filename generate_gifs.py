from sys import argv
import numpy as np
import torch
from sinusoidal_encodings import SinusoidalPositionEmbeddings
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation

def animate(i):

    x=np.arange(EMBEDDING_SIZE)
    y=embeddings[i].detach().cpu().numpy()

    line.set_data(x, y)
    text.set_text('Embedding for position: {}'.format(i))

    return line

if __name__=='__main__':

    EMBEDDING_SIZE=int(argv[1])
    INFORMATION_DENSITY=int(argv[2]) if len(argv)>=3 else 10000
    NUM_POSITION_TO_ENCODE=int(argv[3]) if len(argv)>=4 else 1000

    emb_gen=SinusoidalPositionEmbeddings(EMBEDDING_SIZE, INFORMATION_DENSITY)
    
    t_steps=torch.arange(0, NUM_POSITION_TO_ENCODE)+1
    t_steps=t_steps[:, None]

    embeddings=emb_gen(t_steps).squeeze()

    fig, ax=plt.subplots()
    ax.set_xlim(0, 30) 
    ax.set_ylim(-1.1, 1.1)

    line=ax.plot([], [], lw=3)[0]
    text=ax.text(0.5, 0.9, '', transform=ax.transAxes)

    ani = FuncAnimation(fig, animate, frames=1000, interval=20)
    ani.save('position_embeddings_{}.gif'.format(INFORMATION_DENSITY), writer='pillow')