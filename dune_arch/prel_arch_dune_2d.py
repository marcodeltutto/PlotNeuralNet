import sys
sys.path.append('../')
from pycore.tikzeng import *

xstart = 1280 / 32
ystart = 2048 / 32

start = 2048 / 32

scale = 0.96

ypos = 5

def get_premerge_arch(yoffset=0, xoffset=-5, imagename="/Users/deltutto/Downloads/plane_0_zoom"):
    arch = [
        to_input(imagename, to="("+str(xoffset)+","+str(yoffset)+",0)", width=13, height=8, name="temp"),
        to_Conv("conv1", 2048, 32, offset="(5,"+str(yoffset)+",0)", to="("+str(xoffset)+",0,0)", height=xstart, depth=ystart, width=scale, caption="Initial Convolution" ),

        #---
        to_Conv("convres1", s_filer=2048, n_filer=32, offset="(2,0,0)", to="(conv1-east)", width=6, height=xstart, depth=ystart, color="{rgb:red,1;black,0.08}", caption="2x Residual Block" ),
        to_connection("conv1", "convres1"),
        to_Conv("conv2", 1024, 64, offset="(4,0,0)", to="(convres1-east)", height=xstart/2, depth=ystart/2, width=scale*2, caption="Downsampling" ),
        to_connection( "convres1", "conv2"),
        #---
        to_Conv("convres2", s_filer=1024, n_filer=64, offset="(2,0,0)", to="(conv2-east)", width=6, height=xstart/2, depth=ystart/2, color="{rgb:red,1;black,0.08}", caption=" " ),
        to_connection("conv2", "convres2"),
        to_Conv("conv3", 512, 96, offset="(3,0,0)", to="(convres2-east)", height=xstart/4, depth=ystart/4, width=scale*3 ),
        to_connection( "convres2", "conv3"),
        #---
        to_Conv("convres3", s_filer=512, n_filer=96, offset="(1,0,0)", to="(conv3-east)", width=6, height=xstart/4, depth=ystart/4, color="{rgb:red,1;black,0.08}", caption=" " ),
        to_connection("conv3", "convres3"),
        to_Conv("conv4", 256, 128, offset="(2,0,0)", to="(convres3-east)", height=xstart/8, depth=ystart/8, width=scale*4 ),
        to_connection( "convres3", "conv4"),
        #---
        to_Conv("convres4", s_filer=256, n_filer=128, offset="(1,0,0)", to="(conv4-east)", width=6, height=xstart/8, depth=ystart/8, color="{rgb:red,1;black,0.08}", caption=" " ),
        to_connection("conv4", "convres4"),
        to_Conv("conv5", 128, 160, offset="(2,0,0)", to="(convres4-east)", height=xstart/16, depth=ystart/16, width=scale*5 ),
        to_connection( "convres4", "conv5"),
        # #---
        # to_Conv("convres5", s_filer=128, n_filer=160, offset="(1,0,0)", to="(conv5-east)", width=6, height=xstart/16, depth=ystart/16, color="{rgb:red,1;black,0.08}", caption=" " ),
        # to_connection("conv5", "convres5"),
        # to_Conv("conv6", 64, 192, offset="(1,0,0)", to="(convres5-east)", height=xstart/32, depth=ystart/32, width=scale*6 ),
        # to_connection( "convres5", "conv6"),
        # #---
        # to_Conv("convres6", s_filer=64, n_filer=192, offset="(1,0,0)", to="(conv6-east)", width=6, height=xstart/32, depth=ystart/32, color="{rgb:red,1;black,0.08}", caption=" " ),
        # to_connection( "conv6", "convres6"),
        # to_Conv("conv7", 32, 224, offset="(1,0,0)", to="(convres6-east)", height=xstart/64, depth=ystart/64, width=scale*6 ),
        # to_connection( "convres6", "conv7"),
        # #---
        # # Final layer 2
        # to_Conv("convres_final2", s_filer=32, n_filer=224, offset="(1,0,0)", to="(conv7-east)", width=6, height=xstart/64, depth=ystart/64, color="{rgb:red,1;black,0.08}", caption=" " ),
        # to_connection( "conv7", "convres_final2"),
        # #---
        # # Bottleneck
        # to_Conv("conv_bott", 32, 3, offset="(2,0,0)", to="(convres_final2-east)", height=xstart/32, depth=ystart/32, width=2, caption="Bottleneck" ),
        # to_connection( "convres_final2", "conv_bott"),
        # #---
        # # Av pool 3D
        # to_Pool("pool", offset="(2,0,0)", to="(conv_bott-east)", width=1, height=xstart/32, depth=ystart/32, opacity=0.5, caption="Av 3D Pooling"),
        # to_connection( "conv_bott", "pool"),
        # #---
        # to_SoftMax("criterion", s_filer=2, offset="(2,0,0)", to="(pool-east)", width=1, height=1, depth=1, opacity=0.8, caption="Cross Entropy Loss" ),
        # to_connection( "pool", "criterion"),
        
        # to_end()
        ]
    return arch



def get_postmerge_arch(yoffset=0, xoffset=-5, nplanes=3):
    arch = [
        to_Conv("conv5", 128, 160*nplanes, offset="(5,"+str(yoffset)+",0)", to="("+str(xoffset)+",0,0)", height=xstart/16*nplanes, depth=ystart/16*nplanes, width=scale*5*nplanes ),
        #---
        to_Conv("convres5", s_filer=128, n_filer=160*nplanes, offset="(1,0,0)", to="(conv5-east)", width=6*nplanes, height=xstart/16*nplanes, depth=ystart/16*nplanes, color="{rgb:red,1;black,0.08}", caption=" " ),
        to_connection("conv5", "convres5"),
        to_Conv("conv6", 64, 192*nplanes, offset="(1,0,0)", to="(convres5-east)", height=xstart/32*nplanes, depth=ystart/32*nplanes, width=scale*6*nplanes ),
        to_connection( "convres5", "conv6"),
        #---
        to_Conv("convres6", s_filer=64, n_filer=192*nplanes, offset="(1,0,0)", to="(conv6-east)", width=6*nplanes, height=xstart/32*nplanes, depth=ystart/32*nplanes, color="{rgb:red,1;black,0.08}", caption=" " ),
        to_connection( "conv6", "convres6"),
        to_Conv("conv7", 32, 224*nplanes, offset="(1,0,0)", to="(convres6-east)", height=xstart/64*nplanes, depth=ystart/64*nplanes, width=scale*6*nplanes ),
        to_connection( "convres6", "conv7"),
        #---
        # Final layer 2
        to_Conv("convres_final2", s_filer=32, n_filer=224*nplanes, offset="(1,0,0)", to="(conv7-east)", width=6*nplanes, height=xstart/64*nplanes, depth=ystart/64*nplanes, color="{rgb:red,1;black,0.08}", caption=" " ),
        to_connection( "conv7", "convres_final2"),
        #---
        # Bottleneck
        to_Conv("conv_bott", 32, 3, offset="(2,0,0)", to="(convres_final2-east)", height=xstart/64*nplanes, depth=ystart/64*nplanes, width=1*nplanes, caption="Bottleneck" ),
        to_connection( "convres_final2", "conv_bott"),
        #---
        # Av pool 3D
        to_Pool("pool", offset="(2,0,0)", to="(conv_bott-east)", width=1*nplanes, height=xstart/64*nplanes, depth=ystart/64*nplanes, opacity=0.5, caption="Av 3D Pooling"),
        to_connection( "conv_bott", "pool"),
        #---
        to_SoftMax("criterion", s_filer='$n_c$', offset="(2,0,0)", to="(pool-east)", width=1*nplanes, height=1, depth=1, opacity=0.8, caption="Softmax" ),
        to_connection( "pool", "criterion"),
        
        # to_end()
        ]
    return arch

def get_arch():
    arch = []
    arch.append(to_head( '..' ))
    arch.append(to_cor())
    arch.append(to_begin())
    arch += get_premerge_arch(xoffset=-5, yoffset=15,  imagename="/Users/deltutto/Downloads/plane_0_zoom")
    arch += get_premerge_arch(xoffset=-5, yoffset=0,   imagename="/Users/deltutto/Downloads/plane_1_zoom")
    arch += get_premerge_arch(xoffset=-5, yoffset=-15, imagename="/Users/deltutto/Downloads/plane_2_zoom")
    arch += get_postmerge_arch(xoffset=25, yoffset=0)
    arch.append(
        to_end()
        )
    return arch

def main():
    namefile = str(sys.argv[0]).split('.')[0]
    arch = get_arch()
    # print(arch)
    to_generate(arch, namefile + '.tex' )

if __name__ == '__main__':
    main()
