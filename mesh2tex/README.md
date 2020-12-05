in dir /texture_fields-master/mesh2tex/texnet

added line 4
import lpips

added initializations in line 36 and 37
self.loss_fn_alex = lpips.LPIPS(net='alex') # best forward scores
self.loss_fn_vgg = lpips.LPIPS(net='vgg') # closer to "traditional" perceptual loss, when used for optimization

commented out line 274
#loss = F.l1_loss(img_fake, img_real)

added line 276
loss = self.loss_fn_vgg(img_fake.to('cpu'), img_real.to('cpu'))