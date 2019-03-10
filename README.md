<title>1D GAN demo</title>
<div style="display: flex;justify-content:center; align-items:Center;">
    <video muted src="natgan3.mp4" type="video/mp4" controls="" autoplay="autoplay" loop="loop" width="1000px" height="600px">
    </video>
</div>

Please use Chrome.

Code of 1D GAN is based on <a href=https://github.com/AYLIEN/gan-intro>gan-intro</a>

The two negative distributions are:
``` python
class NegDistribution(object):
    def __init__(self):
        self.mu = 13
        self.sigma = 1

    def sample(self, N):
        samples = np.random.normal(self.mu, self.sigma, N)
        samples.sort()
        return samples

class NegDistribution2(object):
    def __init__(self):
        self.mu = -6
        self.sigma = 2

    def sample(self, N):
        samples = np.random.normal(self.mu, self.sigma, N)
        samples.sort()
        return samples
```

For NAT-GAN on CIFAR 10, code will be available soon, and you can easily implement it by modifying
[AM GAN](https://github.com/ZhimingZhou/AM-GAN2), some important codes are:<br>

``` python
fake_logits = discriminator(fake_datas, num_logits)
fake_logits2 = discriminator(tf.concat([fake_datas, next(neg_gen)[0]], 0), num_logits)

dis_fake_loss = kl_divergence(tf.ones_like(fake_logits, tf.float32) / num_logits, tf.nn.softmax(fake_logits))
dis_fake_loss2 = kl_divergence(tf.ones_like(fake_logits2, tf.float32) / num_logits, tf.nn.softmax(fake_logits2))
```
set class number=10, batchsize=256, and you'd better add negative samples every 10 batches to let model converge better.