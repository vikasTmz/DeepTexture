Code we edited:
mesh2tex/geometry/__init__.py
To allow for a different shape encoder to be used, options needed to be added to the shape encoder dictionary (lines 9 and 10)

mesh2tex/geometry/pointnet.py
We needed to define Pointnet++ as a valid geometry encoder class. The class Pointnet2 (lines 185-218) is a modification of the code
https://github.com/yanx27/Pointnet_Pointnet2_pytorch/blob/master/models/pointnet2_cls_msg.py
Slight modifications were made to allow for compatibility of inputs for this specific use of Pointnet++ 


pointnet_util is a version of the code seen at
https://github.com/yanx27/Pointnet_Pointnet2_pytorch/blob/master/models/pointnet_util.py
This code is necessary for Pointnet++. The only modifications added were on lines 256 and 198 to resolve conflicts with using GPU for computing

mesh2tex/texnet/config.py
Within the function get_models(cfg, dataset=None, device=None),
an IF-ELSE block was added starting at line 44. This statement allowed us to load pre-trained weights when specified by the config file.

mesh2tex/texnet/training.py
added line 4
import lpips

added initializations in line 36 and 37
self.loss_fn_alex = lpips.LPIPS(net='alex') # best forward scores
self.loss_fn_vgg = lpips.LPIPS(net='vgg') # closer to "traditional" perceptual loss, when used for optimization

commented out line 274
#loss = F.l1_loss(img_fake, img_real)

added line 276
loss = self.loss_fn_vgg(img_fake.to('cpu'), img_real.to('cpu'))
