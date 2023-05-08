import argparse
import os
import numpy as np

import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from model import AutoEncoder
from utils.reduce_face_information import trim_face_information

# Input parameters
epochs = 7000
lr = 0.001
#device = 'cpu' # Change from 'cpu' to 'cuda' if you want to use GPU, see next commented line
device = 'cuda'

# Get the name of the object we want to encode as argument to the encoding python file
parser = argparse.ArgumentParser(description='Get the name of the object to encode')
parser.add_argument('-i', type=str)
parser.add_argument('-o', type=str)
args = parser.parse_args()

# Define the input and output paths
input_file_path = args.i
output_file_path = args.o

os.makedirs(os.path.dirname(output_file_path), exist_ok=True)

encoded_mesh_path = os.path.dirname(output_file_path)

#############  DATASET ##########################################
meshes = []

# Load Mesh
with open(input_file_path, "r") as f:
    mesh_object = f.readlines()

mesh_object = [line.strip() for line in mesh_object]
last_vertex_index = -1
for i in range(len(mesh_object)):
    if mesh_object[i].startswith('v '):
        last_vertex_index += 1
    else:
        break
vertices = mesh_object[:last_vertex_index + 1]
faces = mesh_object[last_vertex_index + 1:]

vertices = [item.split()[1:] for item in vertices]
vertices = [float(coordinate) for point in vertices for coordinate in point]
vertices = np.array(vertices)
vertices = vertices.reshape(-1, 3)
num_points = len(vertices)

#To numpy
sample = np.asarray(vertices)
sample = (sample-sample.min())/(sample.max()-sample.min())
np.save('{}/norm_factors.npy'.format(encoded_mesh_path), np.asarray([sample.min(),sample.max()]))

#Normalize sample
meshes.append(sample)

# Create dataloader
data = np.asarray(meshes)
data = torch.from_numpy(data).permute(0,2,1).float().to(device)
my_dataset = TensorDataset(data,data)
trainloader = DataLoader(my_dataset, batch_size=1, shuffle=True)
##################################################################


#############  TRAINING ##########################################
net = AutoEncoder(num_points).to(device)

# Train

criterion = nn.L1Loss()
optimizer = optim.Adam(net.parameters(), lr=lr)
for ind,epoch in enumerate(range(epochs)):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]

        inputs, labels = data

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs, embeddings= net(inputs)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
    print('Loss {}. Epoch {}:{}'.format(running_loss, ind+1,epochs))


print('Saving model and the embeddings')
net = net.half().eval()  # save model in half precision
torch.save(net.decoder.state_dict(), '{}/decoder.ckpt'.format(encoded_mesh_path))
np.save('{}/embeddings.npy'.format(encoded_mesh_path), embeddings.detach().cpu().numpy())

output_file = "{}/faces.txt".format(encoded_mesh_path)
faces = trim_face_information(faces) # you might want to remove this line as it destroys connectivity information
with open(output_file, "w") as output_writer:
    for line in faces:
        output_writer.write(line + "\n")

number_of_vertices_file = "{}/number_of_vertices.txt".format(encoded_mesh_path)
with open(number_of_vertices_file, "w") as output_writer:
    output_writer.write(str(num_points))
