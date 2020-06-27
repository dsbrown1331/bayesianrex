import sys
if len(sys.argv) != 4:
    print("Usage: " + sys.argv[0] + " <input_filename> [output_filename]")
    print("Used to get hex digests of network and potentially transfer them to a new stripped network")
    sys.exit()

ENCODING_DIMS = int(sys.argv[3])
print("stripping network to encode to ", ENCODING_DIMS, "dimensional space")

import hashlib
import torch
import torch.nn as nn
from StrippedNet import EmbeddingNet

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

excess_layers = [('fc_var', {'version': 1}), ('reconstruct1', {'version': 1}), ('reconstruct2', {'version': 1}), ('reconstruct_conv1', {'version': 1}), ('reconstruct_conv2', {'version': 1}), ('reconstruct_conv3', {'version': 1}), ('reconstruct_conv4', {'version': 1}), ('temporal_difference1', {'version': 1}), ('temporal_difference2', {'version': 1}), ('inverse_dynamics1', {'version': 1}), ('inverse_dynamics2', {'version': 1}), ('forward_dynamics1', {'version': 1}), ('forward_dynamics2', {'version': 1}), ('forward_dynamics3', {'version': 1})]

excess_operations = [('softmax', {'version': 1}), ('sigmoid', {'version': 1})]

keys_hit = []

model = torch.load(sys.argv[1], map_location=device)
# t_hash = hashlib.new('ripemd160')
# for layer in excess_layers:
#     if layer[0] + '.bias' not in model:
#         print("Layer not found: " + layer[0])
#         continue
#     keys_hit.append(layer[0] + '.bias')
#     keys_hit.append(layer[0] + '.weight')
#     t_hash.update(bytes(str(model[layer[0] + '.bias']), 'utf-8'))
#     t_hash.update(bytes(str(model[layer[0] + '.weight']), 'utf-8'))
#
# print("Hash of all excess layers: " + t_hash.hexdigest())
#
# t_hash = hashlib.new('ripemd160')
# trex_layers = ['fc2.weight', 'fc2.bias']
# for layer in trex_layers:
#     if layer not in model:
#         print("Layer not found: " + layer)
#         continue
#     t_hash.update(bytes(str(model[layer]), 'utf-8'))
# print("Hash of t-rex layer: " + t_hash.hexdigest())
#
# t_hash = hashlib.new('ripemd160')
# original_layers = ['conv1.weight', 'conv1.bias', 'conv2.weight', 'conv2.bias', 'conv3.weight', 'conv3.bias', 'conv4.weight', 'conv4.bias', 'fc1.weight', 'fc1.bias', 'fc_mu.weight', 'fc_mu.bias']
# for layer in original_layers:
#     if layer not in model:
#         print("Layer not found: " + layer)
#         continue
#     t_hash.update(bytes(str(model[layer]), 'utf-8'))
#
# print("Hash of all original layers: " + t_hash.hexdigest())

#if len(sys.argv) == 4:
#     ENCODING_DIMS = 30
#     class Net(nn.Module):
#         def __init__(self):
#             super().__init__()
#
#             self.conv1 = nn.Conv2d(4, 16, 7, stride=3)
#             self.conv2 = nn.Conv2d(16, 16, 5, stride=2)
#             self.conv3 = nn.Conv2d(16, 16, 3, stride=1)
#             self.conv4 = nn.Conv2d(16, 16, 3, stride=1)
#
#             # This is the width of the layer between the convolved framestack
#             # and the actual latent space. Scales with ENCODING_DIMS
#             intermediate_dimension = min(784, max(64, ENCODING_DIMS*2))
#
#             # Brings the convolved frame down to intermediate dimension just
#             # before being sent to latent space
#             self.fc1 = nn.Linear(784, intermediate_dimension)
#
#             # This brings from intermediate dimension to latent space. Named mu
#             # because in the full network it includes a var also, to sample for
#             # the autoencoder
#             self.fc_mu = nn.Linear(intermediate_dimension, ENCODING_DIMS)
#
#             # This is the actual T-REX layer; linear comb. from ENCODING_DIMS
#             self.fc2 = nn.Linear(ENCODING_DIMS, 1)

net = EmbeddingNet(ENCODING_DIMS)
sd = net.state_dict()
sd.update({k:v for k,v in model.items() if k in net.state_dict()})

torch.save(sd, sys.argv[2])
