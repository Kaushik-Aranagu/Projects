# original coder : https://github.com/D-X-Y/ResNeXt-DenseNet
# added simpnet model 
from __future__ import division

import os, sys, pdb, shutil, time, random, datetime
import argparse
import torch
import torch.optim.lr_scheduler as lr_scheduler
import torch.backends.cudnn as cudnn
import torchvision.datasets as dset
import torchvision.transforms as transforms
from utils import AverageMeter, RecorderMeter, time_string, convert_secs2time
import models
from models.simplenet import simplenet 
import numpy as np
from torch.utils.data.sampler import SubsetRandomSampler
# from tensorboardX import SummaryWriter

model_names = sorted(name for name in models.__dict__
  if name.islower() and not name.startswith("__")
  and callable(models.__dict__[name]))

#print('models : ',model_names)
parser = argparse.ArgumentParser(description='Trains ResNeXt on CIFAR or ImageNet', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--data_path', type=str, default='./data', help='Path to dataset')
parser.add_argument('--dataset', type=str, default='cifar100', choices=['cifar10', 'cifar100', 'imagenet', 'svhn', 'stl10'], help='Choose between Cifar10/100 and ImageNet.')
parser.add_argument('--arch', metavar='ARCH', default='resnet18', choices=model_names, help='model architecture: ' + ' | '.join(model_names) + ' (default: resnext29_8_64)')
# Optimization options
parser.add_argument('--epochs', type=int, default=40, help='Number of epochs to train.')
parser.add_argument('--batch_size', type=int, default=64, help='Batch size.')
parser.add_argument('--learning_rate', type=float, default=0.1, help='The Learning Rate.')
parser.add_argument('--momentum', type=float, default=0.90, help='Momentum.')
parser.add_argument('--decay', type=float, default=0.002, help='Weight decay (L2 penalty).')
parser.add_argument('--schedule', type=int, nargs='+', default=[70,90], help='Decrease learning rate at these epochs.')
parser.add_argument('--gammas', type=float, nargs='+', default=[0.1,0.1], help='LR is multiplied by gamma on schedule, number of gammas should be equal to schedule')
# Checkpoints
parser.add_argument('--print_freq', default=750, type=int, metavar='N', help='print frequency (default: 150)')
parser.add_argument('--save_path', type=str, default='./NewValues', help='Folder to save checkpoints and log.')
parser.add_argument('--resume', default='', type=str, metavar='PATH', help='path to latest checkpoint (default: none)')
parser.add_argument('--start_epoch', default=0, type=int, metavar='N', help='manual epoch number (useful on restarts)')
parser.add_argument('--evaluate', dest='evaluate', action='store_true', help='evaluate model on validation set')
# Acceleration
parser.add_argument('--ngpu', type=int, default=1, help='0 = CPU.')
parser.add_argument('--workers', type=int, default=2, help='number of data loading workers (default: 2)')
# random seed
parser.add_argument('--manualSeed',default=1210, type=int, help='manual seed')
args = parser.parse_args()
args.use_cuda = args.ngpu>0 and torch.cuda.is_available()
if args.manualSeed is None:
  args.manualSeed = random.randint(1, 10000)
random.seed(args.manualSeed)
torch.manual_seed(args.manualSeed)
torch.cuda.manual_seed(args.manualSeed)
if args.use_cuda:
  torch.cuda.manual_seed_all(args.manualSeed)
#speeds things a bit more  
cudnn.benchmark = True
# torch.backends.cudnn.enabled = False
# torch.backends.cudnn.deterministic = True
#asd

level = [None]*5



labels = ["apple","aquarium_fish","baby","bear","beaver","bed","bee","beetle","bicycle","bottle","bowl","boy","bridge","bus","butterfly","camel","can","castle","caterpillar","cattle","chair","chimpanzee","clock","cloud","cockroach","couch","crab","crocodile","cup","dinosaur","dolphin","elephant","flatfish","forest","fox","girl","hamster","house","kangaroo","keyboard","lamp","lawn_mower","leopard","lion","lizard","lobster","man","maple_tree","motorcycle","mountain","mouse","mushroom","oak_tree","orange","orchid","otter","palm_tree","pear","pickup_truck","pine_tree","plain","plate","poppy","porcupine","possum","rabbit","raccoon","ray","road","rocket","rose","sea","seal","shark","shrew","skunk","skyscraper","snail","snake","spider","squirrel","streetcar","sunflower","sweet_pepper","table","tank","telephone","television","tiger","tractor","train","trout","tulip","turtle","wardrobe","whale","willow_tree","wolf","woman","worm","largeCarnivores", "largeOmniAndHerbivores", "mediumSizedMammals", "people", "insects", "smallMammals", "reptiles", "nonInsectInvertebrates", "aquaticMammals", "fish", "flowers", "fruitsAndVegetables", "trees", "vehicles1", "vehicles2", "largeOutdoor", "householdFurniture", "householdElectrical", "foodContainers","bigAnimals", "smallAnimals", "aquaticAnimals", "flora", "naturalOutdoorScenes", "outdoor", "indoor","animals", "nature", "manmade","entity"]

l1 = 1
l2 = 2
l3 = 3
l4 = 5
l5 = 7
lexcept = ["cloud","forest","mountain","plain","sea"]


level[0] = ["entity"]
level[1] = ["animals", "nature", "manmade"]
level[2] = ["bigAnimals", "smallAnimals", "aquaticAnimals", "flora", "naturalOutdoorScenes", "outdoor", "indoor"]
level[3] = ["largeCarnivores", "largeOmniAndHerbivores", "mediumSizedMammals", "people", "insects", "smallMammals", "reptiles", "nonInsectInvertebrates", "aquaticMammals", "fish", "flowers", "fruitsAndVegetables", "trees", "vehicles1", "vehicles2", "largeOutdoor", "householdFurniture", "householdElectrical", "foodContainers","cloud","forest","mountain","plain","sea"]
level[4] = ["apple","aquarium_fish","baby","bear","beaver","bed","bee","beetle","bicycle","bottle","bowl","boy","bridge","bus","butterfly","camel","can","castle","caterpillar","cattle","chair","chimpanzee","clock","cockroach","couch","crab","crocodile","cup","dinosaur","dolphin","elephant","flatfish","fox","girl","hamster","house","kangaroo","keyboard","lamp","lawn_mower","leopard","lion","lizard","lobster","man","maple_tree","motorcycle","mouse","mushroom","oak_tree","orange","orchid","otter","palm_tree","pear","pickup_truck","pine_tree","plate","poppy","porcupine","possum","rabbit","raccoon","ray","road","rocket","rose","seal","shark","shrew","skunk","skyscraper","snail","snake","spider","squirrel","streetcar","sunflower","sweet_pepper","table","tank","telephone","television","tiger","tractor","train","trout","tulip","turtle","wardrobe","whale","willow_tree","wolf","woman","worm"]

def get_level(x):
  if (x in level[0]):
    return 0
  elif (x in level[1]):
    return 1
  elif (x in level[2]):
    return 2
  elif (x in level[3]):
    return 3
  elif (x in level[4]):
    return 4
  else:
    return -1

def get_children(x):
  if x == "entity":
    return ["animals", "nature", "manmade"]
  elif x == "animals":
    return ["bigAnimals", "smallAnimals", "aquaticAnimals"]
  elif x == "nature":
    return ["flora", "naturalOutdoorScenes"]
  elif x == "manmade":
    return ["outdoor", "indoor"]
  elif x == "bigAnimals":
    return ["largeCarnivores", "largeOmniAndHerbivores", "mediumSizedMammals", "people"]
  elif x == "smallAnimals":
    return ["insects", "smallMammals", "reptiles", "nonInsectInvertebrates"]
  elif x == "aquaticAnimals":
    return ["aquaticMammals", "fish"]
  elif x == "flora":
    return ["flowers", "fruitsAndVegetables", "trees"]
  elif x == "naturalOutdoorScenes":
    return ["cloud","forest","mountain","plain","sea"]
  elif x == "outdoor":
    return ["vehicles1", "vehicles2", "largeOutdoor"]
  elif x == "indoor":
    return ["householdFurniture", "householdElectrical", "foodContainers"]
  elif x == "largeCarnivores":
    return ["bear", "leopard", "lion", "tiger", "wolf"]
  elif x ==  "largeOmniAndHerbivores":
    return ["camel", "cattle", "chimpanzee", "elephant", "kangaroo"]
  elif x ==  "mediumSizedMammals":
    return ["fox", "porcupine", "possum", "raccoon", "skunk"]
  elif x ==  "people":
    return ["baby", "boy", "girl", "man", "woman"]
  elif x ==  "insects":
    return ["bee", "beetle", "butterfly", "caterpillar", "cockroach"]
  elif x ==  "smallMammals":
    return ["hamster", "mouse", "rabbit", "shrew", "squirrel"]
  elif x ==  "reptiles":
    return ["crocodile", "dinosaur", "lizard", "snake", "turtle"]
  elif x ==  "nonInsectInvertebrates":
    return ["crab", "lobster", "snail", "spider", "worm"]
  elif x ==  "aquaticMammals":
    return ["beaver", "dolphin", "otter", "seal", "whale"]
  elif x ==  "fish":
    return ["aquarium_fish", "flatfish", "ray", "shark", "trout"]
  elif x ==  "flowers":
    return ["orchid", "poppy", "rose", "sunflower", "tulip"]
  elif x ==  "fruitsAndVegetables":
    return ["apple", "mushroom", "orange", "pear", "sweet_pepper"]
  elif x ==  "trees":
    return ["maple_tree", "oak_tree", "palm_tree", "pine_tree", "willow_tree"]
  elif x ==  "vehicles1":
    return ["bicycle", "bus", "motorcycle", "pickup_truck", "train"]
  elif x ==  "vehicles2":
    return ["lawn_mower", "rocket", "streetcar", "tank", "tractor"]
  elif x ==  "largeOutdoor":
    return ["bridge", "castle", "house", "road", "skyscraper"]
  elif x ==  "householdElectrical":
    return ["clock", "keyboard", "lamp", "telephone", "television"]
  elif x ==  "householdFurniture":
    return ["bed", "chair", "couch", "table", "wardrobe"]
  elif x ==  "foodContainers":
    return ["bottle", "bowl", "can", "cup", "plate"]
  else:
    return None

def get_parent(x):
  l = get_level(x)
  if l == 0: 
    return x
  for i in level[l-1]:
    if x in get_children(i):
      return i
  return None


def get_ancestor(x, l):
  h = get_level(x)
  if l >= h:
    return x

  y = x
  for i in range(h-l):
    y = get_parent(y)
  return y

def get_descendants(x):
  c = get_children(x)
  d = []
  if c is not None:
    for i in c:
      d.extend(get_descendants(i)) 
    return d
  else :
    return [x] 

def tree_loss(x,y):
  l1 = get_level(x)
  l2 = get_level(y)
  l = l2
  if l1 < l2:
    l = l1

  while l >= 0:
    if get_ancestor(x,l) == get_ancestor(y,l):
      break
    else :
      l = l-1

  return l1 + l2 - 2*l

count = 0

def loss(x,y):
  global l1
  global l2
  global l3
  global l4
  global l5

  temp = x * y
  t1 = temp[:,0:l1]
  t2 = temp[:,l1:l1+l2]
  t3 = temp[:,l1+l2:l1+l2+l3]
  t4 = temp[:,l1+l2+l3:l1+l2+l3+l4]
  t5 = temp[:,l1+l2+l3+l4:l1+l2+l3+l4+l5]

  print(x)

  t1 = np.max(t1,axis=1)
  t2 = np.max(t2,axis=1)
  t3 = np.max(t3,axis=1)
  t4 = np.max(t4,axis=1)
  t5 = np.max(t5,axis=1)

  zeros = np.zeros(x.shape[0])
  t1 = max(t1,zeros)
  t2 = max(t2,zeros)
  t3 = max(t3,zeros)
  t4 = max(t4,zeros)
  t5 = max(t5,zeros)

  t = t1+t2+t3+t4+t5

  ave_loss = np.sum(t)

  # print(t1)
  # print(t2)
  # print(t3)
  # print(t4)
  # print(t5)



class BEP_Loss(torch.nn.Module):
    
    def __init__(self):
        super(BEP_Loss,self).__init__()
        
    def forward(self,x,y):
        global l1
        global l2
        global l3
        global l4
        global l5

        temp = x * y
        t1 = temp[:,0:l1]
        t2 = temp[:,l1:l1+l2]
        t3 = temp[:,l1+l2:l1+l2+l3]
        t4 = temp[:,l1+l2+l3:l1+l2+l3+l4]
        t5 = temp[:,l1+l2+l3+l4:l1+l2+l3+l4+l5]

        t1,_ = torch.max(t1,1)
        t2,_ = torch.max(t2,1)
        t3,_ = torch.max(t3,1)
        t4,_ = torch.max(t4,1)
        t5,_ = torch.max(t5,1)

        zero = torch.zeros(x.size()[0])
        zero = zero.cuda()

        t1 = torch.max(t1+1,zero)
        t2 = torch.max(t2+1,zero)
        t3 = torch.max(t3+1,zero)
        t4 = torch.max(t4+1,zero)
        t5 = torch.max(t5+1,zero)

        t = t1+t2+t3+t4+t5

        # clamp = torch.max(temp,zero)
        # # print(clamp[0,:])
        total_loss = torch.sum(t)/x.size()[0]
        # print(y[0])
        # print(y_[0,:])
        # print(temp[0,:])
        # for i in range(130):
        #   s[i] = s[i] + torch.sum(clamp[:,i])/x.size(0)
          # print(s[i])
        # if count % 100 == 0:
        #   print(clamp[0,:])
        #   print(clamp[:,129])
        # count = count + 1
        # print(total_loss)
        return total_loss


val_count = 0
neg_correct = 0
val_count_pos = 0
pos_incorrect = 0
zero_count = 0
one_count = 0
two_count = 0
three_count = 0
four_count = 0

def binaryEncoding(inp,bits):
  out = -np.ones(bits)
  s = bits - 1
  while inp != 0:
    r = inp % 2
    if r == 1:
      out[s] = 1
    inp = inp // 2
    s -= 1 
  return out

def target_trans(target):
  y = -torch.ones(18)

  global l1
  global l2
  global l3
  global l4
  global l5
  # y[target] = 1

  global labels  
  global lexcept
  label = labels[target]

  y = []
  y.extend(binaryEncoding(level[0].index(get_ancestor(label,0)),l1))
  y.extend(binaryEncoding(level[1].index(get_ancestor(label,1)),l2))
  y.extend(binaryEncoding(level[2].index(get_ancestor(label,2)),l3))
  y.extend(binaryEncoding(level[3].index(get_ancestor(label,3)),l4))
  y.extend(binaryEncoding(target,l5))

  y = torch.FloatTensor(y)
  # for j in range(30):
  #   if label in get_descendants(labels[100+j]):
  #     y[100+j] = 1
  # print(y)
  return y,target

def main():
  # Init logger
  if not os.path.isdir(args.save_path):
    os.makedirs(args.save_path)

  # Get timestamp 
  time_stamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')   

  # Random seed used 
  print("Random Seed: {0}".format(args.manualSeed))
  # Python Version used  
  print("python version : {}".format(sys.version.replace('\n', ' ')), log)
  # Torch Version used
  print("torch  version : {}".format(torch.__version__), log)
  # Cudnn Version used
  print("cudnn  version : {0}".format(torch.backends.cudnn.version()))

  # Path for the dataset. If not present, it is downloaded
  if not os.path.isdir(args.data_path):
    os.makedirs(args.data_path)


  # Preparing dataset
  if args.dataset == 'cifar10':
    mean = [x / 255 for x in [125.3, 123.0, 113.9]]
    std = [x / 255 for x in [63.0, 62.1, 66.7]]
  elif args.dataset == 'cifar100':
    mean = [x / 255 for x in [129.3, 124.1, 112.4]]
    std = [x / 255 for x in [68.2, 65.4, 70.4]]
  else:
    assert False, "Unknow dataset : {}".format(args.dataset)





  # Additional dataset transforms like padding, crop and flipping of images
  train_transform = transforms.Compose(
    [ transforms.RandomCrop(32, padding=4),transforms.RandomHorizontalFlip(), transforms.ToTensor(),
     transforms.Normalize(mean, std)])
  test_transform = transforms.Compose(
    [transforms.CenterCrop(32), transforms.ToTensor(), transforms.Normalize(mean, std)])


  # Loading dataset from the path to data
  if args.dataset == 'cifar10':
    train_data = dset.CIFAR10(args.data_path, train=True, transform=train_transform, download=True)
    test_data = dset.CIFAR10(args.data_path, train=False, transform=test_transform, download=True)
    num_classes = 10
  elif args.dataset == 'cifar100':
    train_data = dset.CIFAR100(args.data_path, train=True, transform=train_transform, target_transform=target_trans, download=True)
    test_data = dset.CIFAR100(args.data_path, train=False, transform=test_transform, target_transform=target_trans, download=True)
    num_classes = 18
  else:
    assert False, 'Does not support dataset : {}'.format(args.dataset)

  # Splitting the training dataset into train and val sets
  num_train = len(train_data)
  indices = list(range(num_train))
  # Number of images in the validation set (Here, 10000 out of the total 50000 images)
  split = int(10000)

  np.random.seed(args.manualSeed)
  np.random.shuffle(indices)

  train_idx, valid_idx = indices[split:], indices[:split]
  train_sampler = SubsetRandomSampler(train_idx)
  valid_sampler = SubsetRandomSampler(valid_idx)


  # Loading the data into loaders
  train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, sampler=train_sampler,
                         num_workers=args.workers, pin_memory=True)

  val_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, sampler=valid_sampler,
                         num_workers=args.workers, pin_memory=True)

  test_loader = torch.utils.data.DataLoader(test_data, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.workers, pin_memory=True)


  # Print the architecture of the model being used
  print("=> creating model '{}'".format(args.arch), log)


  # Initializing the model. Initializes using weights. 
  net = simplenet(classes=18)

  # Using GPUs for the Neural Network
  if args.ngpu > 0:
    net = torch.nn.DataParallel(net, device_ids=list(range(args.ngpu)))

  # Define loss function (criterion)
  criterion = BEP_Loss()
  # Define the optimizer used in the laerning algorithm (optimizer)
  optimizer = torch.optim.Adadelta(pars, lr=0.1, rho=0.9, eps=1e-3, weight_decay=0)

  # This is the list of all the parameters. Used when all the layers are to be trained 
  # (Comment for training only the last layer and uncomment the following command)
  pars = net.parameters()

  # Classifier has only two layers (net.module.classifier and net.module.classifier_)
  # This is the list of parameters in the last two layers of the network. Used when only layers in the classifier layers are to be trained
  # (Uncomment for training only the classifier layers and the rest of the layers of the network frozen)
  # pars = list(net.module.classifier.parameters()) + list(net.module.classifier_.parameters())

  # Classifier has three layers (net.module.classifier and net.module.classifier_ and net.module.classifier2_)
  # This is the list of parameters in the last three layers of the network. Used when only the layers in the classifier are to be trained
  # (Uncomment for training only the classifier layers and the rest of the layers of the network frozen)
  # pars = list(net.module.classifier.parameters()) + list(net.module.classifier_.parameters()) + list(net.module.classifier2_.parameters())


  states_settings = {
                     'optimizer': optimizer.state_dict()
                     }


  # Epochs after which when the learning rate is to be changed 
  milestones = [15,25,35]
  # Defines how the leraning rate is changed at the above mentioned epochs.
  # 'gamma' is the factor by which the learning rate is reduced 
  scheduler = lr_scheduler.MultiStepLR(optimizer, milestones, gamma=0.1)

  # Put the network and the other data on the CUDA device
  if args.use_cuda:
    net.cuda()
    criterion.cuda()
    print('__Number CUDA Devices:', torch.cuda.device_count())

  # A structure to record different results while training and evaluating 
  recorder = RecorderMeter(args.epochs)

  # If this argument is given, the network loads weights from the checkpoint file given in the arguments
  if args.resume:
    if os.path.isfile(args.resume):
      print("=> loading checkpoint '{}'".format(args.resume))
      checkpoint = torch.load(args.resume)

      # If no GPU is being used, uncomment the following line
      # checkpoint = torch.load(args.resume,map_location='cpu')

      # When loading from a checkpoint with only features saved, uncomment the following line else comment it
      # net.module.features.load_state_dict(checkpoint['state_dict'])

      # When loading from a checkpoint with all the parameters saved, uncomment the following line else comment it. 
      net.load_state_dict(checkpoint['state_dict'])

      print("=> loaded checkpoint '{}' (epoch )" .format(args.resume))
  else:
    print("=> did not use any checkpoint for {} model".format(args.arch))

  # Opening a log, None when not using it
  log = None

  # Evaluating a model on training, testing and validation sets
  if args.evaluate:
    print("Train data : ")
    acc,loss,tloss = validate(train_loader, net, criterion, log)
    print(loss, acc, tloss)
    print("Test data : ")
    acc,loss, tloss = validate(test_loader, net, criterion, log)
    print(loss,acc, tloss)
    print("Validation data : ")
    acc,loss, tloss = validate(val_loader, net, criterion, log)
    print(loss,acc, tloss)
    return

  # Main loop

  # Start timer
  start_time = time.time()
  # Structure to record time for each epoch
  epoch_time = AverageMeter()

  # Starts training from epoch 0
  args.start_epoch = 0
  best_loss = 1000000
  best_acc = 0
  log = None

  # Loop for training for each epoch
  for epoch in range(args.start_epoch, args.epochs):

    # Get learning rate for the epoch
    current_learning_rate = adjust_learning_rate(optimizer, epoch, args.gammas, args.schedule)
    current_learning_rate = float(scheduler.get_lr()[-1])

    scheduler.step()

    # Calcluate the time for the remaing number of epochs
    need_hour, need_mins, need_secs = convert_secs2time(epoch_time.avg * (args.epochs-epoch))
    need_time = '[Need: {:02d}:{:02d}:{:02d}]'.format(need_hour, need_mins, need_secs)

    # Train the network on the training data and return accuracy, training loss and the t-loss on the training dataset
    train_acc, train_los, train_tloss = train(train_loader, net, criterion, optimizer, epoch, log)

    # Print after each epoch the remaining time, epoch number, max accuracy recorded on validation set so far and other details
    print('\n==>>{:s} [Epoch={:03d}/{:03d}] {:s} [learning_rate={:.6f}]'.format(time_string(), epoch, args.epochs, need_time, current_learning_rate) \
                + ' [Best : Accuracy={:.2f}, Error={:.2f}]'.format(recorder.max_accuracy(False), 100-recorder.max_accuracy(False)))


    # Evaluate on the validation data and update the recorded values
    val_acc,   val_los, val_tloss  = validate(val_loader, net, criterion, log)
    is_best = recorder.update(epoch, train_tloss, train_acc, val_tloss, val_acc, )

    # Save checkpoint after every 30 epochs
    if epoch % 40 == 39:
        save_checkpoint({
          'epoch': epoch ,
          'arch': args.arch,
          # Save the whole network with the following line uncommented. If only features are to be saved, comment it.
          'state_dict': net.state_dict(),
          # Save only the features layers of the network with the following line uncommented. If thw whole network is to be saved, comment it.
          # 'state_dict': net.module.features.state_dict(),
          'recorder': recorder,
          'optimizer' : optimizer.state_dict(),
          # Name of the ckeckpoint file to be saved to
        }, False, args.save_path, 'beploss_frozen_full_{0}_2layer_200_50.ckpt'.format(epoch), time_stamp)


    # measure elapsed time
    epoch_time.update(time.time() - start_time)
    start_time = time.time()
    recorder.plot_curve( os.path.join(args.save_path, 'training_plot_finetuning_beploss_{0}_1.png'.format(args.manualSeed)) )

  # writer.close()
  # End loop

  # Evaluate and print the results on training, testing and validation data
  test_acc,   test_los, tl   = validate(train_loader, net, criterion, log)  
  print("Train accuracy : ")
  print(test_acc)
  print(test_los)
  print(tl)

  test_acc,   test_los, tl   = validate(val_loader, net, criterion, log)  
  print("Val accuracy : ")
  print(test_acc)
  print(test_los)
  print(tl)

  test_acc,   test_los, tl   = validate(test_loader, net, criterion, log)  
  print("Test accuracy : ")
  print(test_acc)
  print(test_los)
  print(tl)


  # log.close()



# train function (forward, backward, update)
def train(train_loader, model, criterion, optimizer, epoch, log):

  # Structures to store various data
  batch_time = AverageMeter()
  data_time = AverageMeter()
  losses = AverageMeter()
  top1 = AverageMeter()
  top5 = AverageMeter()
  tloss = AverageMeter()

  # switch to train mode
  model.train()

  # Record the start time
  end = time.time()

  # Looping on each batch
  for i, (input, target) in enumerate(train_loader):

    # y is the new representation of the label for the image
    # target_trans in line 212 gives the representation of y
    y = target[1]
    # target is the true label of the image
    target = target[0]

    # measure data loading time
    data_time.update(time.time() - end)

    # Putting the data on the GPU
    if args.use_cuda:
      target = target.cuda()
      input = input.cuda()
      y = y.cuda()
    input_var = torch.autograd.Variable(input, requires_grad=False)
    target_var = torch.autograd.Variable(target, requires_grad=False)

    # compute output
    output = model(input_var)
    loss = criterion(output, target_var)

    # Dummy
    tl = 0

    # measure accuracy and record loss
    prec1, prec5 = accuracy(output.data, y, topk=(1, 5))
    losses.update(loss.data.item(), input.size(0))
    top1.update(prec1.item(), input.size(0))
    top5.update(prec5.item(), input.size(0))
    tloss.update(tl, input.size(0))

    # compute gradient and do SGD step
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # measure elapsed time
    batch_time.update(time.time() - end)
    end = time.time()

    # Print output at print frquency
    if i % args.print_freq == 0:
      print('  Epoch: [{:03d}][{:03d}/{:03d}]   '
            'Time {batch_time.val:.3f} ({batch_time.avg:.3f})   '
            'Data {data_time.val:.3f} ({data_time.avg:.3f})   '
            'Loss {loss.val:.4f} ({loss.avg:.4f})   '
            'Prec@1 {top1.val:.3f} ({top1.avg:.3f})   '
            'Prec@5 {top5.val:.3f} ({top5.avg:.3f})   '.format(
            epoch, i, len(train_loader), batch_time=batch_time,
            data_time=data_time, loss=losses, top1=top1, top5=top5) + time_string(), log)
  print('  **Train** Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f} Error@1 {error1:.3f}'.format(top1=top1, top5=top5, error1=100-top1.avg))

  # Return the average accuracy, loss and t-loss on the training set  
  # t-loss here is dummy. We get the actual t-loss on the train dataset after the whole training is done
  return top1.avg, losses.avg, tloss.avg

def validate(val_loader, model, criterion, log):
  losses = AverageMeter()
  top1 = AverageMeter()
  tloss = AverageMeter()
  top5 = AverageMeter()

  # switch to evaluate mode
  model.eval()

  # Loop on the dataset
  for i, (input, target) in enumerate(val_loader):

    # y is the new representation of the label for the image
    # target_trans in line 212 gives the representation of y
    y = target[1]
    # target is the true label of the image
    target = target[0]

    # Load onto GPU
    if args.use_cuda:
      target = target.cuda()
      input = input.cuda()
      y = y.cuda()
    input_var = torch.autograd.Variable(input, requires_grad = False)
    target_var = torch.autograd.Variable(target, requires_grad = False)

    # compute output
    output = model(input_var)
    loss = criterion(output, target_var)
    # Compute t-loss
    tl = tLoss(output.data,y)

    # measure accuracy and record loss
    prec1, prec5 = accuracy(output.data, y, topk=(1, 5))
    losses.update(loss.data.item(), input.size(0))
    top1.update(prec1.item(), input.size(0))
    top5.update(prec5.item(), input.size(0))
    tloss.update(tl,input.size(0))

  print('  **Test** Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f} Error@1 {error1:.3f}'.format(top1=top1, top5=top5, error1=100-top1.avg))

  # Return accuracy, loss and t-loss on the dataset evaluated
  return top1.avg, losses.avg, tloss.avg


# def print(print_string, log):
#   print("{}".format(print_string))
#   log.write('{}\n'.format(print_string))
#   log.flush()

def save_checkpoint(state, is_best, save_path, filename, timestamp=''):
  filename = os.path.join(save_path, filename)
  torch.save(state, filename)
  # if is_best:
  #   bestname = os.path.join(save_path, 'model_best_{0}.pth.tar'.format(timestamp))
  #   shutil.copyfile(filename, bestname)

def adjust_learning_rate(optimizer, epoch, gammas, schedule):
  """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
  lr = args.learning_rate
  assert len(gammas) == len(schedule), "length of gammas and schedule should be equal"
  for (gamma, step) in zip(gammas, schedule):
    if (epoch >= step):
      lr = lr * gamma
    else:
      break
  for param_group in optimizer.param_groups:
    param_group['lr'] = lr
  return lr


def bep_to_target(output):
  sign = -torch.sign(output)
  out = sign[:,11:18]
  out = (out+1)/2
  # print(out)
  # print(output[:,11:18])
  out = 64*out[:,0:1] + 32*out[:,1:2] + 16*out[:,2:3] + 8*out[:,3:4] + 4*out[:,4:5] + 2*out[:,5:6] + out[:,6:7]
  # print(out)
  out = out.type(torch.FloatTensor)
  return out


# Get topk accuracy for an output of the model
def accuracy(output, target, topk=(1,)):
  """Computes the precision@k for the specified values of k"""

  maxk = max(topk)
  batch_size = target.size(0)

  # We get the true class label of the instance from the binary encoding used in target_trans function in line 322
  pred = bep_to_target(output)
  pred = pred.t()
  target = target.type(torch.FloatTensor)
  correct = pred.eq(target.view(1, -1).expand_as(pred))

  res = []
  for k in topk:
    correct_k = correct[:k].view(-1).float().sum(0)
    res.append(correct_k.mul_(100.0 / batch_size))
  return res

def b2i(v,h):
  cons = [1,2,6,23,99]
  l = [1,2,3,5,7]
  d = l[h]
  v = -torch.sign(v)
  v = (v+1)/2
  x = 1
  o = 0
  for i in range(d):
    o = o + x * v[d-i-1]
    x = 2 * x

  if o > cons[h]:
    o = o - 2**(d-1)
  return o.type(torch.LongTensor)

# Calculate t-loss for an output
def tLoss(output,target):
  batch_size = target.size(0)
  num_classes = output.size(1)
  h = 4
  val = None
  pred = None
  loss = 0
  global labels
  global level

  # Loop for each example in the batch
  for i in range(batch_size):
    t = target[i]
    # Go searching bottom-up in the hierarchy tree
    h = 4
    while h >= 0:
      # Clone the output array each time for checking at every level in the bottom-up search 
      values = output.clone().detach()
      # At each level in the bottom-up traversal, we take the minimum of the outputs of nodes
      # (a particular range in the totat set of output nodes of the model) and if it is 
      # greater than the threshold at that level, we break and get class index at level from 
      # the signs of those outputs (b2i function defined above)
      if h == 0:
        values = values[i,0:l1]
        val,_ = torch.min(torch.abs(values),0)        
        if val >= 0:
          pred = b2i(values,h)
          pred_label = level[h][pred]
          break
      if h == 1:
        values = values[i,l1:l1+l2]
        val,_ = torch.min(torch.abs(values),0)        
        if val >= 0:
          pred = b2i(values,h)
          pred_label = level[h][pred]
          break
      if h == 2:
        values = values[i,l1+l2:l1+l1+l2+l3]
        val,_ = torch.min(torch.abs(values),0)        
        if val >= 0.5:
          pred = b2i(values,h)
          pred_label = level[h][pred]
          break
      if h == 3:
        values = values[i,l1+l2+l3:l1+l2+l3+l4]
        val,_ = torch.min(torch.abs(values),0)        
        if val >= 0.5:
          pred = b2i(values,h)
          pred_label = level[h][pred]
          break
      if h == 4:
        values = values[i,l1+l2+l3+l4:l1+l2+l3+l4+l5]
        val,_ = torch.min(torch.abs(values),0)        
        if val >= 0.25:
          pred = b2i(values,h)
          pred_label = labels[pred]
          break
      h = h - 1

    # tree_loss() function gives the tree distance between the two nodes
    # Total loss for the batch is calculated
    loss = loss + tree_loss(labels[t],pred_label)

  loss = loss/batch_size

  # Average loss for the batch is returned
  return loss

if __name__ == '__main__':
  main()
