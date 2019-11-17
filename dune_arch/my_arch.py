import sys
sys.path.append('../')
from pycore.tikzeng import *

xstart = 1280 / 32
ystart = 2048 / 32

start = 2048 / 32

scale = 0.03

ypos = 5

# defined your arch
arch = [
    to_head( '..' ),
    to_cor(),
    to_begin(),
    to_input("/Users/deltutto/Downloads/plane_0_zoom", to="(-5,"+str(ypos)+",0)", width=13, height=8, name="temp"),
    to_Conv("conv1", 2048, 32, offset="(0,"+str(ypos)+",0)", to="(0,0,0)", height=xstart, depth=ystart, width=32*scale, caption="Initial Convolution" ),

    #---
    to_Conv("convres1", s_filer=2048, n_filer=32, offset="(2,0,0)", to="(conv1-east)", width=6, height=xstart, depth=ystart, color="{rgb:red,1;black,0.08}", caption="Residual Block" ),
    to_connection("conv1", "convres1"),
    to_Conv("conv2", 256, 64, offset="(4,0,0)", to="(convres1-east)", height=xstart/2, depth=ystart/2, width=32*scale*2, caption="Downsampling" ),
    to_connection( "convres1", "conv2"),
    #---
    to_Conv("convres2", s_filer=256, n_filer=64, offset="(2,0,0)", to="(conv2-east)", width=6, height=xstart/2, depth=ystart/2, color="{rgb:red,1;black,0.08}", caption=" " ),
    to_connection("conv2", "convres2"),
    to_Conv("conv3", 128, 96, offset="(3,0,0)", to="(convres2-east)", height=xstart/4, depth=ystart/4, width=32*scale*3 ),
    to_connection( "convres2", "conv3"),
    #---
    to_Conv("convres3", s_filer=128, n_filer=96, offset="(2,0,0)", to="(conv3-east)", width=6, height=xstart/4, depth=ystart/4, color="{rgb:red,1;black,0.08}", caption=" " ),
    to_connection("conv3", "convres3"),
    to_Conv("conv4", 64, 128, offset="(2,0,0)", to="(convres3-east)", height=xstart/8, depth=ystart/8, width=32*scale*4 ),
    to_connection( "convres3", "conv4"),
    #---
    to_Conv("convres4", s_filer=64, n_filer=128, offset="(1,0,0)", to="(conv4-east)", width=6, height=xstart/8, depth=ystart/8, color="{rgb:red,1;black,0.08}", caption=" " ),
    to_connection("conv4", "convres4"),
    to_Conv("conv5", 32, 160, offset="(1,0,0)", to="(convres4-east)", height=xstart/16, depth=ystart/16, width=32*scale*5 ),
    to_connection( "convres4", "conv5"),
    #---
    to_Conv("convres5", s_filer=32, n_filer=160, offset="(1,0,0)", to="(conv5-east)", width=6, height=xstart/16, depth=ystart/16, color="{rgb:red,1;black,0.08}", caption=" " ),
    to_connection("conv5", "convres5"),
    to_Conv("conv6", 16, 192, offset="(1,0,0)", to="(convres5-east)", height=xstart/32, depth=ystart/32, width=32*scale*6 ),
    to_connection( "convres5", "conv6"),
    #---
    # Final layer 1
    to_Conv("convres_final1", s_filer=16, n_filer=192, offset="(1,0,0)", to="(conv6-east)", width=6, height=xstart/32, depth=ystart/32, color="{rgb:red,1;black,0.08}", caption=" " ),
    to_connection( "conv6", "convres_final1"),
    #---
    # Final layer 2
    to_Conv("convres_final2", s_filer=16, n_filer=192, offset="(1,0,0)", to="(convres_final1-east)", width=6, height=xstart/32, depth=ystart/32, color="{rgb:red,1;black,0.08}", caption=" " ),
    to_connection( "convres_final1", "convres_final2"),
    #---
    # Bottleneck
    to_Conv("conv_bott", 16, 2, offset="(2,0,0)", to="(convres_final2-east)", height=xstart/32, depth=ystart/32, width=2, caption="Bottleneck" ),
    to_connection( "convres_final2", "conv_bott"),
    #---
    # Av pool 3D
    to_Pool("pool", offset="(2,0,0)", to="(conv_bott-east)", width=1, height=xstart/32, depth=ystart/32, opacity=0.5, caption="Av 3D Pooling"),
    to_connection( "conv_bott", "pool"),
    #---
    to_SoftMax("criterion", s_filer=2, offset="(2,0,0)", to="(pool-east)", width=1, height=1, depth=1, opacity=0.8, caption="Cross Entropy Loss" ),
    to_connection( "pool", "criterion"),
    
    to_end()
    ]

def main():
    namefile = str(sys.argv[0]).split('.')[0]
    to_generate(arch, namefile + '.tex' )

if __name__ == '__main__':
    main()
