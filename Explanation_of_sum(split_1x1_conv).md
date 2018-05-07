"""
Created on Sat May  5 19:04:09 2018

@author: sungkyun
"""

        Illustration of padding and omission for split_1x1 conv. (exactly same with padded-2x1 dil-conv.) 
        We assume an input of 8 elements as in figure2 of the paper.
        
            <First Block> with recep_sz = 8
            input   =  [1, 2, 3, 4, 5, 6, 7, 8]
            split_L =  [0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 3, 4]  <-- left-zpad_sz = recep_sz = 8  , right-omit = recep_sz/2 = 4
            split_R =  [0, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8]  <-- left-zpad_sz = recep_sz/2 = 4
            
            sum_LR[0:4] = [0,0,0,0] * w_R + [0,0,0,0] * w_L, <-- *, where w_ is weights of 1x1 Conv. Hereafter, we omit w_.
            sum_LR[1:5] = [0,0,0,0] * w_R + [0,0,0,1] * w_L,
            ...
            sum_LR[8:end]= [1,2,3,4] * w_R +[5,6,7,8] * w_L
            
            sum_LR      = [00, 00, 00, 00, 01, 02, 03, 04, 15, 26, 37, 48]
            
            <Second Block> with recep_sz = 4
            input       = [00, 00, 00, 00, 01, 02, 03, 04, 15, 26, 37, 48]  
            split_L     = [00, 00, 00, 00, 01, 02, 03, 04, 15, 26]  <-- right-omit = recep_sz/2 = 2
            split_R     = [00, 00, 01, 02, 03, 04, 15, 26, 37, 48]  <-- left-omit  = recep_sz/2 = 2
            sum_LR      = [0000, 0000, 0001, 0002, 0103, 0204, 0315, 0426, 1537, 2648]
            
            <Third Block> with recep_sz = 2
            input       = [0000, 0000, 0001, 0002, 0103, 0204, 0315, 0426, 1537, 2648]
            split_L     = [0000, 0000, 0001, 0002, 0103, 0204, 0315, 0426, 1537]  <-- right-omit = recep_sz/2 = 1
            split_R     = [0000, 0001, 0002, 0103, 0204, 0315, 0426, 1537, 2648]  <-- left-omit  = recep_sz/2 = 1 
            sum_LR      = [00000000, 00000001, 00010002, 00020103, 01030204, 02040315, 03150426, 04261537, 15372648]
                           
            <Final>
            right-omit(1) 
            = [00000000, 00000001, 00010002, 00020103, 01030204, 02040315, 03150426, 04261537]
            
            will predict:
              [1,        2       , 3       , 4       , 5       , 6       , 7       , 8]
              
            Note that all the matrices are transposed (in code, input = batch x ch x 8 ).
            In summary, the required paddings and omissions are: 
                <First Block>
                L: left_zpad(input, recep_sz), right_omit(recep_sz/2)
                R: left-zpad(recep_sz/2)
                <Following Blocks>
                recep_sz = recep_sz/2
                L: right_omit(recep_sz/2)
                R: left_omit(recep_sz/2)
                <Final stage before FC layer>
                right_omit(1)
