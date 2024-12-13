from torchstat import stat
import torchvision.models as models

mode = models.vgg13()
stat(mode, (3,224,224))
