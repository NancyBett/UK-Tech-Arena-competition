import argparse
import os
import numpy as np
import torch

from model import Decoder

# Decoding parameters
device = torch.device('cpu')  # Change from 'cpu' to 'cuda' if you want to use GPU, see next commented line
# device = torch.device('cuda')

# Get the name of the object we want to decode as argument to the decoding python file
parser = argparse.ArgumentParser(description='Get the name of the object to decode')
parser.add_argument('-i', type=str)
parser.add_argument('-o', type=str)
args = parser.parse_args()

# Define the input and output paths
input_file_path = args.i
output_file_path = args.o

os.makedirs(os.path.dirname(output_file_path), exist_ok=True)

encoded_mesh_path = os.path.dirname(input_file_path)
decoder_path = '{}/decoder.ckpt'.format(encoded_mesh_path)
embeddings_path = '{}/embeddings.npy'.format(encoded_mesh_path)
norm_factors_path = '{}/norm_factors.npy'.format(encoded_mesh_path)
faces_path = '{}/faces.txt'.format(encoded_mesh_path)
number_of_vertices_file = "{}/number_of_vertices.txt".format(encoded_mesh_path)
decoded_mesh_path = output_file_path

# Load the number of vertices of the mesh from file
with open(number_of_vertices_file) as f:
    num_points = int(f.readlines()[0])

# Load model
model = Decoder(num_points=num_points).to(device)
model.load_state_dict(torch.load(decoder_path, map_location=device))
model = model.eval()
# Load embedding
x = np.load(embeddings_path)

# Load normalizing factors
norm_factors = np.load(norm_factors_path)

# Inference
x = torch.from_numpy(x).to(device)
y = model(x)

# Denormalize
y = y.permute(0,2,1)
y = y.detach().cpu().numpy()
y = y*(norm_factors[1]-norm_factors[0]) + norm_factors[0]

# Convert vertex data to list
decoded_mesh_vertices = y[0]
decoded_mesh_vertices = np.float16(decoded_mesh_vertices)
decoded_mesh_vertices = decoded_mesh_vertices.reshape(-1).tolist()
decoded_mesh_vertices = [round(item, 6) for item in decoded_mesh_vertices]

# Write decoded mesh vertex data to file
output_mesh = []
for i in range(len(decoded_mesh_vertices) // 3):
    vertex = "v {} {} {} \n".format(str(decoded_mesh_vertices[3*i]), str(decoded_mesh_vertices[3*i+1]), str(decoded_mesh_vertices[3*i+2]))
    output_mesh.append(vertex)

with open(decoded_mesh_path, "w") as output_writer:
    for line in output_mesh:
        output_writer.write(line)

# Add saved face information to decoded mesh file
with open(faces_path) as faces_reader:
    faces = faces_reader.readlines()

with open(decoded_mesh_path, "a") as output_writer:
    for line in faces:
        output_writer.write(line)
