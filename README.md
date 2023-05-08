# Example AI Template

This repository is an example template of what a project could look like. Our demo code uses a neural network approach to auto-encode the vertices of an input mesh. We train a neural network for each mesh.

## Creating Your Repository

Create a new repository for your project within this GitHub organization. You can use the `Use this template` button on GitHub to create a copy of this repository for your team. Use your team name for the repository name and make sure all your team members have access to your repository. To add them go to `Settings` then `Collaborators and teams`.

Then set up your new repository locally. You can clone it or use GitHub Desktop. Both options are available on the `Code` dropdown menu. To clone your repository and checkout the submodules use the following Git commands.

```
git clone [YourRepositoryGitURL]
cd [YourRepositoryName]
git submodule update --init --recursive
```

Now you are ready to start building your solution.

## Building the Solution

Run the `scripts/build_ai.sh` script to install the dependencies.

```
bash scripts/build_ai.sh
```

This is just a template, you need to write your own code to compress and decompress the `.obj` or `.glTF` sample files.

If you want to create your own build script you can copy `scripts/build_ai.sh` outside the scripts folder (you cannot modify the scripts folder). Make sure to also update the `.github/workflows/ci.yml` file to run your new script on GitHub Actions.

## Testing the Solution

The `scripts/test_ai.sh` script runs all the sample models through the encoder and decoder.

```
bash scripts/test_ai.sh
```

This command will execute two python files: `encode.py` and `decode.py`, iterating over all meshes in `sample-models`.

1) The first file, `encode.py`, will take a mesh as input from the directory `sample-models` and produce its encodings, together with neural network parameters per mesh and also normalisation constants per mesh. It will also save non-vertex information and the number of vertices of the mesh in the `test` folder. Those files are saved in the directory `test/encoded`. Each mesh should save 5 files required for decoding: `decoder.ckpt`,  `embeddings.npy`, `faces.txt`, `norm_factors.npy` and `number_of_vertices.txt`.

The command below produces the encoding for one given input mesh:

```
$ python3 encode.py -i <some_mesh_object> -o <some_output_directory>
```

To use a GPU for encodinng and decoding set `device` to `cuda` in the `encode.py` and/or `decode.py` files.

```
device = torch.device('cuda')
```

2) The second file, `decode.py`, will load the encoded mesh, together with the neural network parameters, the normalisation parameters of the mesh, the non-vertex data, and number of vertices, and output a reconstructed mesh. This mesh will be saved in the directory `test/decoded`.

The command below produces the decoding for one mesh:

```
$ python3 decode.py -i <some_mesh_directory> -o <some_output_mesh_object>
```

The packages required to run this code can be found in the file `requirements.txt`.

This code serves only as a proof of concept for AI model compression. The contestants are free to use it or build on top of it, though this is not required.

The code parameters, model architecture, network input and output data structures, or general approach are not optimised. Very likely, different and better solutions exist. We highly encourage participants to explore innovative ideas.

Please note that this code trains a neural network per mesh. An interesting different direction would be to have a unique neural network which can encode and decode all meshes.

Important: For each mesh, you might need to run the training script more than once to get the network train well on that mesh. Normally, if you see the training loss decreasing early, then the training round will be successful. On the other hand, if the loss does not decrease, you likely need to restart the encoding and decoding for that object.

The test script measures the **compression ratio**, **decompression time** and **image quality** for each sample model and then calculates a **weighted average** score.

It writes the compressed and decompressed files to the `test` folder.

And it logs the values for each model to the log files.

- `test/compression.log` the format is *Encoded/Failed FilePath CompressionRatio*
- `test/decompression.log` the format is *Decoded/Failed FilePath DecompressionTime*
- `test/quality.log` the format is *Quality/Failed FilePath PSNR*

## Support

Check the [FAQs](https://github.com/UKTechArena/.github/blob/main/FAQ.md)

If you still have any questions or experience any problems please reach out to us on the [#support](https://app.slack.com/client/T0447CNHDT2/C046F57C0C8) channel on Slack.
